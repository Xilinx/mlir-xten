//===- LinalgToDse -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "xten/Transform/LinalgToDsePass.h"

#include <iostream>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "linalg-to-dse"

using namespace mlir;
using namespace xilinx::xten;

namespace {
// This could be done better but is not worth the variadic template trouble.
template <typename Matcher>
static unsigned countMatches(FuncOp f, Matcher &matcher) {
  unsigned count = 0;
  f.walk([&count, &matcher](Operation *op) {
    if (matcher.match(op))
      ++count;
  });
  return count;
}

// Match a zero-padding operation like:
//    %21 = linalg.pad_tensor %20 low[0, 0, 1, 1] high[0, 0, 1, 1]  {
//    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
//      linalg.yield %cst : f32
//    } : tensor<1x1024x13x13xf32> to tensor<1x1024x15x15xf32>
class M_padFpZero {
public:
  bool match(Operation *op) const {
    // check we have the right operator
    auto padOp = dyn_cast<linalg::PadTensorOp>(op);
    if (!padOp)
      return false;

    // get the yielded constant value, if any
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return false;

    // check the yielded constant value is FP zero
    FloatAttr padFPConst;
    if (!matchPattern(padValue, m_Constant(&padFPConst)))
      return false;
    bool retval = padFPConst.getValue().isZero();
    return retval;
  };
};

// Match a bias-copy operation like:
//    %24 = linalg.generic {...} ins(%cst_17 : tensor<125xf32>) outs(%23 :
//    tensor<1x125x7x7xf32>) { ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
//      linalg.yield %arg1 : f32
//    } -> tensor<1x125x7x7xf32>
class M_biasCopy {
public:
  bool match(Operation *op) const {
    // check we have the right operator
    auto genOp = dyn_cast<linalg::GenericOp>(op);
    if (!genOp)
      return false;
    if (genOp.inputs().size() != 1)
      return false;

    auto &yieldBlock = genOp.getRegion().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(yieldBlock.getTerminator());
    if (!yieldOp || yieldOp.values().size() != 1)
      return false;

    // When we have a single block, yielding a single block arg, this
    // should be suited to bias copying. Note: there is a small risk
    // that the loop is copying the output to itself - I didn't know
    // how to check this and deferred looking into it as it seems so
    // useless (probably famous last words).
    Value yieldValue = yieldOp.values().front();
    return yieldValue.isa<BlockArgument>();
  };
};

std::string as_str(Attribute const &r) {
  std::string name;
  llvm::raw_string_ostream OS(name);
  r.print(OS);
  return name;
}

int64_t to_int(OpFoldResult const &val) {
  Attribute attr = val.dyn_cast<Attribute>();
  return attr.cast<IntegerAttr>().getInt();
}

std::vector<int64_t> as_array(SmallVectorImpl<OpFoldResult> const &low,
                              SmallVectorImpl<OpFoldResult> const &high) {
  // low/high specify padding as NCHW
  return {to_int(low[2]), to_int(low[3]), to_int(high[2]), to_int(high[3])};
}

std::vector<int64_t> as_nhwc_array(RankedTensorType const &t) {
  auto shape = t.getShape(); // in NCHW format
  return {shape[0], shape[2], shape[3], shape[1]};
}

std::vector<int64_t> wgts_to_kernel_array(RankedTensorType const &t) {
  auto shape = t.getShape(); // in FCHW format
  return {shape[2], shape[3], shape[0]};
}

std::vector<int64_t>
size_and_wgts_to_kernel_array(DenseIntElementsAttr const &size,
                              RankedTensorType const &wgts) {
  auto shape = wgts.getShape(); // in FCHW format
  auto vals = llvm::to_vector<2>(size.getValues<int64_t>());
  return {vals[0], vals[1], shape[0]};
}

std::vector<int64_t> as_array(DenseIntElementsAttr const &t) {
  auto vals = llvm::to_vector<2>(t.getValues<int64_t>());
  return {vals.begin(), vals.end()};
}

std::vector<int64_t> as_2d_array(DenseIntElementsAttr const &t) {
  auto vals = llvm::to_vector<2>(t.getValues<int64_t>());
  return {vals[0], vals[1]};
}

struct NextName {
  int nextNum = 1;
  std::string getNext() {
    return std::string("l" + std::to_string(nextNum++));
  }
};

struct OutputOp {
  llvm::json::Object object;

  OutputOp(StringRef type) : object({{"type", type}}) {}
  void addConv2dPart(Attribute const &name, std::vector<int64_t> inDim,
                     std::vector<int64_t> filterDim,
                     std::vector<int64_t> strideDim,
                     std::vector<int64_t> padDim) {
    object.insert({"node_name", as_str(name)});
    object.insert({"in_dim", inDim});
    object.insert({"filter_dim", filterDim});
    object.insert({"stride_dim", strideDim});
    object.insert({"pad_dim", padDim});
  }

  void add(std::string key, std::string value) {
    object.insert({key, value});
  }
  void add(std::string key, std::vector<int64_t> value) {
    object.insert({key, value});
  }
};

struct LinalgToDsePass : public LinalgToDseBase<LinalgToDsePass> {

public:
  LinalgToDsePass() = default;
  LinalgToDsePass(LinalgToDsePass const &other)
      : LinalgToDseBase<LinalgToDsePass>(other) {}

  Option<std::string> LinalgToDseFilename{
      *this, "output-file",
      llvm::cl::desc("Output filename to write data for DSE"),
      llvm::cl::init("-")};

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto module = getOperation();
    auto f = module.lookupSymbol<mlir::FuncOp>("forward");
    assert(f.getNumArguments() == 1 && "forward func must have 1 args");

    // Ops that do not need to be implemented.
    auto patternConstant = m_Op<arith::ConstantOp>();
    auto patternInit = m_Op<linalg::InitTensorOp>();

    // To collect potential sub-operators of layers.
    auto patternPadZero = M_padFpZero();
    llvm::SetVector<Operation *> pads;
    auto patternBias = M_biasCopy();
    llvm::SetVector<Operation *> biases;

    // To collect terminal operators of layers.
    struct NodeResult {
      Value result;
      std::string id;
    };
    std::map<Operation *, NodeResult> inputToResult;

    // To collect the return value.
    auto patternRet = m_Op<ReturnOp>();
    Value retVal;

    NextName nextName;

    llvm::json::Object nodes;
    auto walkResult = f.walk<WalkOrder::PreOrder>([&](Operation *op)
                                                      -> WalkResult {
      if (op == f) // don't analyze the function op itself
        return WalkResult::advance();

      if (patternConstant.match(op) || patternInit.match(op)) // ignore
        return WalkResult::skip();
      if (patternPadZero.match(op)) {
        pads.insert(op);
        return WalkResult::skip();
      }
      if (patternBias.match(op)) {
        biases.insert(op);
        return WalkResult::skip();
      }
      if (patternRet.match(op)) {
        retVal = op->getOperand(0);
        return WalkResult::skip();
      }

      if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
        auto *outOp = conv.outputs()[0].getDefiningOp();
        biases.remove(outOp);
        auto *inOp = op->getOperand(0).getDefiningOp();
        auto name = nextName.getNext();
        OutputOp output("Conv2D");
        mlir::RankedTensorType c2d_inDim = nullptr;
        std::vector<int64_t> c2d_padDim;

        if (pads.remove(inOp)) {
          auto *padOp = inOp;
          inOp = padOp->getOperand(0).getDefiningOp();
          auto pad = dyn_cast<linalg::PadTensorOp>(padOp);

          c2d_inDim = pad.getSourceType();
          c2d_padDim = as_array(pad.getMixedHighPad(), pad.getMixedLowPad());
        } else {
          c2d_inDim =
              conv.inputs()[0].getType().dyn_cast<mlir::RankedTensorType>();
          c2d_padDim = {0, 0, 0, 0};
        }

        auto c2d_name = conv->getAttr("layer_name");
        auto c2d_filterDim =
            conv.inputs()[1].getType().dyn_cast<mlir::RankedTensorType>();

        output.addConv2dPart(
            c2d_name,
            as_nhwc_array(c2d_inDim),
            wgts_to_kernel_array(c2d_filterDim),
            as_2d_array(conv.strides()),
            c2d_padDim);

        inputToResult[inOp] = {op->getResult(0), name};
        nodes.insert({name, std::move(output.object)});

        return WalkResult::skip();
      }

      if (auto conv = dyn_cast<linalg::Conv2DLreluMaxpoolOp>(op)) {
        auto *inOp = op->getOperand(0).getDefiningOp();
        if (pads.remove(inOp)) {
          auto *padOp = inOp;
          inOp = padOp->getOperand(0).getDefiningOp();
          auto name = nextName.getNext();
          inputToResult[inOp] = {op->getResult(0), name};
          auto pad = dyn_cast<linalg::PadTensorOp>(padOp);

          OutputOp output("Conv2D_LeakyRelu_MaxPool2D");
          output.addConv2dPart(
              /*name=*/conv->getAttr("layer_name"),
              /*inDim=*/as_nhwc_array(pad.getSourceType()),
              /*filterDim=*/
              wgts_to_kernel_array(conv.inputs()[1]
                                       .getType()
                                       .dyn_cast<mlir::RankedTensorType>()),
              /*strideDim=*/as_2d_array(conv.stride()),
              /*padDim=*/as_array(pad.getMixedHighPad(), pad.getMixedLowPad()));
          output.add("postp_filter_dim",
                     size_and_wgts_to_kernel_array(
                         conv.mp_kernel_size(),
                         conv.inputs()[1]
                             .getType()
                             .dyn_cast<mlir::RankedTensorType>()));
          output.add("postp_stride_dim", as_2d_array(conv.mp_stride()));
          output.add("postp_pad_dim", as_array(conv.mp_padding()));
          nodes.insert({name, std::move(output.object)});
        } else {
          op->emitError("The Xilinx fused conv2d operators are expected to be "
                        "paired with a padding operator");
          return WalkResult::interrupt();
        }

        return WalkResult::skip();
      }

      if (auto conv = dyn_cast<linalg::Conv2DLreluOp>(op)) {

        auto *inOp = op->getOperand(0).getDefiningOp();
        if (pads.remove(inOp)) {
          auto *padOp = inOp;
          inOp = padOp->getOperand(0).getDefiningOp();
          auto name = nextName.getNext();
          OutputOp output("Conv2D_LeakyRelu");
          inputToResult[inOp] = {op->getResult(0), name};
          auto pad = dyn_cast<linalg::PadTensorOp>(padOp);
          output.addConv2dPart(
              /*name=*/conv->getAttr("layer_name"),
              /*inDim=*/as_nhwc_array(pad.getSourceType()),
              /*filterDim=*/
              wgts_to_kernel_array(conv.inputs()[1]
                                       .getType()
                                       .dyn_cast<mlir::RankedTensorType>()),
              /*strideDim=*/as_2d_array(conv.stride()),
              /*padDim=*/as_array(pad.getMixedHighPad(), pad.getMixedLowPad()));
          nodes.insert({name, std::move(output.object)});
        } else {
          op->emitError("The Xilinx fused conv2d operators are expected to be "
                        "paired with a padding operator");
          return WalkResult::interrupt();
        }
        return WalkResult::skip();
      }

      op->emitError("unmatched operator");
      return WalkResult::interrupt();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
    if (!(pads.empty() && biases.empty())) {
      llvm::for_each(pads, [&](Operation *op) {
        op->emitError("unmatched pad operator");
      });
      llvm::for_each(biases, [&](Operation *op) {
        op->emitError("unmatched generic broadcast operator");
      });
      return signalPassFailure();
    }

    // Build the connected list of operators.
    Value prevOutput = f.getArgument(0);
    llvm::json::Array graph;
    while (prevOutput != retVal) {
      auto nextResultIt = inputToResult.find(prevOutput.getDefiningOp());
      if (nextResultIt == inputToResult.end()) {
        if (prevOutput == f.getArgument(0))
          f->emitError("The argument is not used by another operator.");
        else
          prevOutput.getDefiningOp()->emitError(
              "The operator result is not used by another operator.");
      }
      auto &nextResult = nextResultIt->second;
      graph.push_back(nextResult.id);
      prevOutput = nextResult.result;
    }

    llvm::json::Object final(
        {{"nodes", std::move(nodes)}, {"graph", std::move(graph)}});
    llvm::json::OStream out(llvm::outs(), 2);
    out.value(std::move(final));
    out.flush();
    llvm::outs() << "\n";
  }
};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToDsePass() {
  return std::make_unique<LinalgToDsePass>();
}

} // namespace xten
} // namespace xilinx
