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

using mlir::matchers::m_Any;
using mlir::matchers::m_Val;

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

std::string as_str(OpFoldResult const &r) {
  if (auto c = getConstantIntValue(r)) {
    std::stringstream str;
    str << *c;
    return str.str();
  }
  return "<unknown>";
}

std::string as_str(SmallVectorImpl<OpFoldResult> const &low,
                   SmallVectorImpl<OpFoldResult> const &high) {
  std::stringstream str;
  std::string split;
  // low/high specify padding as NCHW
  str << "[" << as_str(low[2]) << ", " << as_str(low[3]) << ", "
      << as_str(high[2]) << ", " << as_str(high[3]) << "]";
  return str.str();
}

std::string as_nhwc_str(RankedTensorType const &t) {
  std::stringstream str;
  auto shape = t.getShape(); // in NCHW format
  str << "[" << shape[0] << ", " << shape[2] << ", " << shape[3] << ", "
      << shape[1] << "]";
  return str.str();
}

std::string wgts_to_kernel_str(RankedTensorType const &t) {
  std::stringstream str;
  auto shape = t.getShape(); // in FCHW format
  str << "[" << shape[2] << ", " << shape[3] << ", " << shape[0] << "]";
  return str.str();
}

std::string size_and_wgts_to_kernel_str(DenseIntElementsAttr const &size,
                                        RankedTensorType const &wgts) {
  std::stringstream str;
  auto shape = wgts.getShape(); // in FCHW format
  auto vals = llvm::to_vector<2>(size.getValues<int64_t>());
  str << "[" << vals[0] << ", " << vals[1] << ", " << shape[0] << "]";
  return str.str();
}

std::string as_2d_str(DenseIntElementsAttr const &t,
                             std::string prefix = {}) {
  std::stringstream str;
  auto vals = llvm::to_vector<2>(t.getValues<int64_t>());
  str << "[" << prefix << vals[0] << ", " << vals[1] << "]";
  return str.str();
}

struct LinalgToDsePass
    : public LinalgToDseBase<LinalgToDsePass> {

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

    auto patternPadZero = M_padFpZero();
    auto patternConstant = m_Op<arith::ConstantOp>();
    auto patternRet = m_Op<ReturnOp>();
    auto patternInit = m_Op<linalg::InitTensorOp>();
    auto patternConvLreluMaxpool = m_Op<linalg::Conv2DLreluMaxpoolOp>();
    auto patternConvLrelu = m_Op<linalg::Conv2DLreluOp>();
    auto patternConv = m_Op<linalg::Conv2DNchwFchwOp>();
    auto patternBias = M_biasCopy();
    llvm::SetVector<Operation *> pads;
    llvm::SetVector<Operation *> biases;
    f.walk<WalkOrder::PreOrder>([&patternConstant, &patternRet, &patternInit,
                                 &patternPadZero, &patternConvLreluMaxpool,
                                 &patternConvLrelu, &patternConv, &patternBias,
                                 &pads, &biases,
                                 &f](Operation *op) -> WalkResult {
      if (op == f) // don't analyze the function op itself
        return WalkResult::advance();

      if (patternConstant.match(op) || patternRet.match(op) ||
          patternInit.match(op)) // ignore
        return WalkResult::skip();
      if (patternPadZero.match(op)) {
        pads.insert(op);
        return WalkResult::skip();
      }
      if (patternBias.match(op)) {
        biases.insert(op);
        return WalkResult::skip();
      }
      if (patternConv.match(op)) {
        auto *outOp = op;
        if (biases.count(outOp)) {
          // We know that this is the bias copy for this conv2d.
          // No further information is needed.
          biases.remove(outOp);
        }
        auto *inOp = op->getOperand(0).getDefiningOp();
        if (pads.count(inOp)) {
          pads.remove(inOp);
          auto pad = dyn_cast<linalg::PadTensorOp>(inOp);
          auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op);
          llvm::outs() << "Conv2DOp:\n";
          llvm::outs() << " - node_name=" << conv->getAttr("layer_name")
                       << "\n";
          llvm::outs() << " - in_dim=" << as_nhwc_str(pad.getSourceType())
                       << "\n";
          llvm::outs() << " - filter_dim="
                       << wgts_to_kernel_str(
                              conv.inputs()[1]
                                  .getType()
                                  .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - stride_dim=" << as_2d_str(conv.strides()) << "\n";
          llvm::outs() << " - pad_dim="
                       << as_str(pad.getMixedHighPad(), pad.getMixedLowPad())
                       << "\n";
        } else {
          auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op);
          llvm::outs() << "Conv2DOp:\n";
          llvm::outs() << " - node_name=\"" << conv->getAttr("layer_name")
                       << "\"\n";
          llvm::outs() << " - in_dim="
                       << as_nhwc_str(conv.inputs()[1]
                                          .getType()
                                          .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - filter_dim="
                       << wgts_to_kernel_str(
                              conv.inputs()[1]
                                  .getType()
                                  .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - stride_dim=" << as_2d_str(conv.strides()) << "\n";
          llvm::outs() << " - pad_dim=[0, 0, 0, 0]\n";
        }
        return WalkResult::skip();
      }
      if (patternConvLreluMaxpool.match(op)) {
        auto *inOp = op->getOperand(0).getDefiningOp();
        if (pads.count(inOp)) {
          pads.remove(inOp);
          auto pad = dyn_cast<linalg::PadTensorOp>(inOp);
          auto conv = dyn_cast<linalg::Conv2DLreluMaxpoolOp>(op);
          llvm::outs() << "Conv2DLreluMaxpoolOp:\n";
          llvm::outs() << " - node_name=\"" << conv->getAttr("layer_name")
                       << "\"\n";
          llvm::outs() << " - in_dim=" << as_nhwc_str(pad.getSourceType())
                       << "\n";
          llvm::outs() << " - filter_dim="
                       << wgts_to_kernel_str(
                              conv.inputs()[1]
                                  .getType()
                                  .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - stride_dim=" << as_2d_str(conv.stride()) << "\n";
          llvm::outs() << " - pad_dim="
                       << as_str(pad.getMixedHighPad(), pad.getMixedLowPad())
                       << "\n";
          llvm::outs() << " - postp_filter_dim="
                       << size_and_wgts_to_kernel_str(
                              conv.mp_kernel_size(),
                              conv.inputs()[1]
                                  .getType()
                                  .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - postp_stride_dim=" << as_2d_str(conv.mp_stride())
                       << "\n";
          llvm::outs() << " - postp_pad_dim="
                       << as_2d_str(conv.mp_padding(), /*prefix=*/"0, 0, ")
                       << "\n";
        } else
          op->emitError("The Xilinx fused conv2d operators are expected to be "
                        "paired with a padding operator");

        return WalkResult::skip();
      }
      if (patternConvLrelu.match(op)) {
        auto *inOp = op->getOperand(0).getDefiningOp();
        if (pads.count(inOp)) {
          pads.remove(inOp);
          auto pad = dyn_cast<linalg::PadTensorOp>(inOp);
          auto conv = dyn_cast<linalg::Conv2DLreluOp>(op);
          llvm::outs() << "Conv2DLreluOp:\n";
          llvm::outs() << " - node_name=\"" << conv->getAttr("layer_name")
                       << "\"\n";
          llvm::outs() << " - in_dim=" << as_nhwc_str(pad.getSourceType())
                       << "\n";
          llvm::outs() << " - filter_dim="
                       << wgts_to_kernel_str(
                              conv.inputs()[1]
                                  .getType()
                                  .dyn_cast<mlir::RankedTensorType>())
                       << "\n";
          llvm::outs() << " - stride_dim=" << as_2d_str(conv.stride()) << "\n";
          llvm::outs() << " - pad_dim="
                       << as_str(pad.getMixedHighPad(), pad.getMixedLowPad())
                       << "\n";
        } else
          op->emitError("The Xilinx fused conv2d operators are expected to be "
                        "paired with a padding operator");
        return WalkResult::skip();
      }

      op->emitError("unmatched operator");
      return WalkResult::advance();
    });

    llvm::for_each(
        pads, [&](Operation *op) { op->emitError("unmatched pad operator"); });
    llvm::for_each(biases, [&](Operation *op) {
      op->emitError("unmatched generic broadcast operator");
    });
    // auto ifm = m_Val(f.getArgument(0));
    // M_conv2d(M_padZero(ivm));
    // {
    //   auto p0 = m_Op<linalg::Conv2DLreluMaxpoolOp>();
 
    //   llvm::outs() << "Pattern Conv2dLreluMaxpoolOp(*) matched "
    //                << countMatches(f, p0) << " times\n";
    // }
    // {
    //   auto p0 = m_Op<linalg::PadTensorOp>();
 
    //   llvm::outs() << "Pattern PadTensorOp(ifm) matched " << countMatches(f, p0)
    //                << " times\n";
    // }
    // {
    //   auto p0 = M_padFpZero();
 
    //   llvm::outs() << "Pattern M_padFpZero(ifm) matched " << countMatches(f, p0)
    //                << " times\n";
    // }

    // forward.walk([&](Operation *op) {
    //   llvm::outs() << " forward:";
    //   op->print(llvm::outs());
    //   llvm::outs() << "\n";
    // });
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
