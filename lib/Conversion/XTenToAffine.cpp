//===- XTenToAffine.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "xten/Conversion/XTenToAffinePass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>

#define DEBUG_TYPE "xten-to-affine-pass"

using namespace mlir;
using namespace xilinx;
using namespace mlir::torch;

namespace xilinx {
namespace xten {

std::vector<uint64_t> Conv2dLoopOrder;
std::vector<uint64_t> Conv2dTileSizes;

}
}

namespace {

LogicalResult
lowerConv2d(Operation *op) {
  return failure();
  // auto conv2d = cast<xten::Conv2dOp>(op);
  // auto loc = conv2d.getLoc();
  // //auto module = op->getParentOfType<ModuleOp>();
  // OpBuilder builder(op);
  // edsc::ScopedContext scope(builder, loc);

  // Value zero = constIndex(0);

  // std::vector<Operation*> erasedOps;

  // TensorType resultTy = conv2d.getResult().getType().cast<TensorType>();
  // MemRefType memRefResultTy = MemRefType::get(resultTy.getShape(),
  //                                             resultTy.getElementType(),
  //                                             {}, 0);

  // auto result = builder.create<memref::AllocOp>(loc, memRefResultTy);
  // Value input = MemRefTypeCast(builder, conv2d.input());
  // Value weight = MemRefTypeCast(builder, conv2d.weight());
  // Value bias = MemRefTypeCast(builder, conv2d.bias());

  // // get layer name to add to loop nest labels 
  // std::string layer_name; 
  // if (!op->getAttrs().empty()) {
  //     for (NamedAttribute attr : op->getAttrs()) {
  //         std::string attr_key = attr.first.c_str();
  //         if (attr_key.compare("name") == 0) {
  //             llvm::raw_string_ostream output(layer_name);
  //             attr.second.print(output);
  //             layer_name.pop_back();
  //             layer_name.erase(layer_name.begin());
  //         }
  //         layer_name.append(".");
  //     }
  // }


  // SmallVector<int64_t,2> stride, padding, dilation;
  // matchPattern(conv2d.stride(), Torch::m_TorchConstantIntList(stride));
  // matchPattern(conv2d.padding(), Torch::m_TorchConstantIntList(padding));
  // matchPattern(conv2d.dilation(), Torch::m_TorchConstantIntList(dilation));

  // auto ctx = op->getContext();

  // edsc::MemRefBoundsCapture vOutput(result);
  // edsc::MemRefBoundsCapture vInput(input);
  // edsc::MemRefBoundsCapture vWeight(weight);

  // edsc::intrinsics::MemRefIndexedValue iOutput(result);
  // edsc::intrinsics::MemRefIndexedValue iInput(input);
  // edsc::intrinsics::MemRefIndexedValue iWeight(weight);

  // if (bias.getType().isa<ShapedType>()) {
  //   edsc::intrinsics::MemRefIndexedValue iBias(bias);
  //   auto loop_body = [&](ValueRange ivs) {
  //     Value ofm_batch = ivs[0], ofm_channel = ivs[1];
  //     Value ofm_row = ivs[2], ofm_col = ivs[3];
  //     iOutput(ofm_batch, ofm_channel, ofm_row, ofm_col) = iBias(ofm_channel);
  //   };
  //   edsc::affineLoopNestBuilder({zero, zero, zero, zero},
  //                               {vOutput.ub(0), vOutput.ub(1), vOutput.ub(2), vOutput.ub(3)},
  //                               {1,1,1,1}, loop_body);
  //   auto a = cast<AffineForOp>(--builder.getInsertionPoint());
  //   a->setAttr("affine_opt_label", StringAttr::get(op->getContext(), layer_name + "xten.conv2d_op_bias"));
  // }

  // //
  // // Construct a conv2d loop nest
  // //
  // Operation *conv_mul = nullptr;
  // Operation *conv_add = nullptr;

  // uint64_t batch_hw = 4;
  // uint64_t ifm_channels_hw = 32;
  // uint64_t ofm_channels_hw = 32;
  // if (xten::Conv2dTileSizes.size()) {
  //   batch_hw = xten::Conv2dTileSizes[0];
  //   ofm_channels_hw = xten::Conv2dTileSizes[1];
  //   ifm_channels_hw = xten::Conv2dTileSizes[2];
  // }

  // FuncOp hwPlaceholderOp;

  // // batch=0, ofm=1, ifm=2
  // std::vector<uint64_t> loopOrder{0,1,2,3,4};
  // if (xten::Conv2dLoopOrder.size()) {
  //   loopOrder = xten::Conv2dLoopOrder;
  // }

  // auto body_builder = [&](ValueRange ivs) {
  //   Value ofm_batch = ivs[ 1 ], ofm_channel = ivs[ 2 ], ifm_channel = ivs[ 3 ];
  //   Value ofm_row = ivs[0], ofm_col = ivs[4];

  //   std::vector<Type> retTy;
  //   std::vector<Value> ofm_indices{ofm_batch, ofm_channel, ofm_row, ofm_col};
  //   auto ident_4d = AffineMap::getMultiDimIdentityMap(4, ctx);
  //   edsc::affineLoopNestBuilder(
  //   {zero, zero},
  //   {vWeight.ub(2), vWeight.ub(3)},
  //   {1,1}, [&](ValueRange ivs3) {
  //     Value kx = ivs3[0], ky = ivs3[1];
  //     // ifm_row = ofm_row * stride + (ky - pad)
  //     // ifm_col = ofm_col * stride + (kx - pad);
  //     auto ifm_row_col_expr = getAffineDimExpr(0, ctx) *
  //       getAffineConstantExpr(stride[0], ctx) +
  //       (getAffineDimExpr(1, ctx) - getAffineConstantExpr(padding[0], ctx));
  //     auto ifm_row_col_map = AffineMap::get(2, 0, ifm_row_col_expr);
  //     std::vector<Value> ifm_row_ops{ofm_row, ky};
  //     std::vector<Value> ifm_col_ops{ofm_col, kx};
  //     auto b = edsc::ScopedContext::getBuilderRef();
  //     auto ifm_row = b.create<AffineApplyOp>(loc, ifm_row_col_map, ifm_row_ops);
  //     auto ifm_col = b.create<AffineApplyOp>(loc, ifm_row_col_map, ifm_col_ops);

  //     std::vector<Value> w_indices{ofm_channel, ifm_channel, ky, kx};
  //     std::vector<Value> ifm_indices{ofm_batch, ifm_channel, ifm_row, ifm_col};

  //     auto partial_load = b.create<AffineLoadOp>(loc, result, ident_4d, ofm_indices);
  //     auto input_load = b.create<AffineLoadOp>(loc, input, ident_4d, ifm_indices);
  //     auto weight_load = b.create<AffineLoadOp>(loc, weight, ident_4d, w_indices);
  //     conv_mul = b.create<MulFOp>(loc, input_load, weight_load);
  //     conv_add = b.create<AddFOp>(loc, conv_mul->getResult(0), partial_load);
  //     /*auto partial_store =*/ b.create<AffineStoreOp>(loc, conv_add->getResult(0), result, ident_4d, ofm_indices);
  //   });
  // };

  // // upper bounds
  // std::vector<Value> ubs;
  // for (auto i : loopOrder) {
  //   std::vector<Value> bounds{vOutput.ub(0), vOutput.ub(1), vInput.ub(1), vOutput.ub(2), vOutput.ub(3)};
  //   ubs.push_back(bounds[i]);
  // }
  // edsc::affineLoopNestBuilder({zero, zero, zero, zero, zero},
  //                             ubs,
  //                             {1,1,1,1,1},
  //                             body_builder);

  // auto afo = cast<AffineForOp>(--builder.getInsertionPoint());
  // afo->setAttr("affine_opt_label", StringAttr::get(op->getContext(), layer_name + "xten.conv2d_op"));

  // LLVM_DEBUG(llvm::outs() << "\nInitial Conv2d loop nest:\n");
  // LLVM_DEBUG(op->getBlock()->print(llvm::outs()));
  // auto tensor_result = TensorTypeCast(builder,result);
  // op->getResult(0).replaceAllUsesWith(tensor_result);
  return success();
}

/// Lower conv2d
class XTenConv2dOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dOpConversion(MLIRContext *context)
      : ConversionPattern(xten::Conv2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    return failure();
  }
};

class XTenMMOpConversion : public ConversionPattern {
public:
  explicit XTenMMOpConversion(MLIRContext *context)
      : ConversionPattern(xten::MMOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    // auto mmult = cast<xten::MMOp>(op);
    // auto loc = mmult.getLoc();

    // using namespace edsc;
    // edsc::ScopedContext scope(rewriter, loc);

    // rewriter.setInsertionPointAfter(op);

    // Type resultTy = mmult.getResult().getType();
    // TensorType tensorResultTy = resultTy.cast<TensorType>();
    // MemRefType memRefResultTy = MemRefType::get(tensorResultTy.getShape(),
    //                                             tensorResultTy.getElementType(),
    //                                             {}, 0);

    // //Value result = rewriter.create<aten::AcapAllocOp>(loc, memRefResultTy, ArrayRef<Value>{});
    // Value result = rewriter.create<memref::AllocOp>(loc, memRefResultTy);
    // Value lhs = MemRefTypeCast(rewriter, operands[0]);
    // Value rhs = MemRefTypeCast(rewriter, operands[1]);
    // Value zero = constIndex(0);

    // edsc::MemRefBoundsCapture vRes(result), vLHS(lhs), vRHS(rhs);
    // edsc::intrinsics::MemRefIndexedValue iRes(result), iLHS(lhs), iRHS(rhs);

    // if (vRes.rank() != 2) {
    //   return failure();
    // }

    // Value i, j;
    // Value M(vRes.ub(0));
    // Value N(vRes.ub(1));

    // affineLoopNestBuilder({zero, zero}, {M, N},
    //                       {1,1}, [&] (ValueRange ivs) {
    //                         Value i = ivs[0]; Value j = ivs[1];
    //                         std::vector<Value> indices{i, j};
    //                         auto ident_2d = AffineMap::getMultiDimIdentityMap(2, op->getContext());
    //                         auto load0 = rewriter.create<AffineLoadOp>(loc, lhs, ident_2d, indices);
    //                         auto load1 = rewriter.create<AffineLoadOp>(loc, rhs, ident_2d, indices);
    //                         auto add = rewriter.create<MulFOp>(loc, load0, load1);
    //                         /*auto store =*/ rewriter.create<AffineStoreOp>(loc, result, add, ident_2d, indices);
    //                       });

    // auto tensor_result = TensorTypeCast(rewriter,result);
    // rewriter.replaceOp(op, {tensor_result});
    // return success();
    return failure();
  }
};

class XTenAddConstantOpConversion : public ConversionPattern {
public:
  explicit XTenAddConstantOpConversion(MLIRContext *context)
      : ConversionPattern(xten::AddConstantOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto addOp = cast<xten::AddConstantOp>(op);
    auto loc = addOp.getLoc();

    rewriter.setInsertionPointAfter(op);

    Torch::BaseTensorType tensorType =
        op->getOperand(0).getType().cast<Torch::BaseTensorType>();

    auto sizes = tensorType.getSizes();
    auto dtype = tensorType.getDtype();
    auto rank = sizes.size();

    MemRefType memRefResultTy = MemRefType::get(sizes, dtype, {}, 0);

    LLVM_DEBUG(llvm::outs() << "\nIn XTenAddConstantOpConversion\n");
    LLVM_DEBUG(op->getBlock()->print(llvm::outs()));

    Value result = rewriter.create<memref::AllocOp>(loc, memRefResultTy);
    Value lhs = xten::MemRefTypeCast(rewriter, operands[0]);

    bool isFloatOp = isa<Torch::ConstantFloatOp>(operands[1].getDefiningOp());

    SmallVector<int64_t, 4> lbs(rank, 0);
    SmallVector<int64_t, 4> steps(rank, 1);

    buildAffineLoopNest(
        rewriter, loc, lbs, sizes, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          SmallVector<Value, 4> indices;
          auto ident = AffineMap::getMultiDimIdentityMap(ivs.size(),
                                                         builder.getContext());
          auto load = builder.create<AffineLoadOp>(loc, lhs, ident, ivs);
          std::vector<uint64_t> index{};
          Value add = nullptr;
          if (isFloatOp) {
            auto c = cast<Torch::ConstantFloatOp>(operands[1].getDefiningOp())
                         .value();
            auto ty = rewriter.getF32Type();
            auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
            add = builder.create<mlir::arith::AddFOp>(
                loc, load, builder.create<ConstantOp>(loc, ty, add_const));
          } else {
            Torch::ConstantIntOp op;
            auto c =
                cast<Torch::ConstantIntOp>(operands[1].getDefiningOp()).value();
            auto ty = rewriter.getIntegerType(32);
            auto add_const = rewriter.getI32IntegerAttr(c.getZExtValue());
            add = builder.create<mlir::arith::AddIOp>(
                loc, load, builder.create<ConstantOp>(loc, ty, add_const));
          }
          builder.create<AffineStoreOp>(loc, add, result, ident, ivs);
        });

    for (auto it = Block::iterator(op), ie = rewriter.getInsertionPoint();
         it != ie; ++it) {
      if (auto afo = dyn_cast<AffineForOp>(it))
        afo->setAttr("affine_opt_label",
                     StringAttr::get(op->getContext(), "affine_opt"));
    }

    auto tensor_result =
        xten::TensorTypeCast(rewriter, result, op->getResult(0).getType());
    rewriter.replaceOp(op, {tensor_result});
    return success();
  }
};

template <class T>
class XTenBinaryOpConversion : public ConversionPattern {

public:
  XTenBinaryOpConversion(StringRef rootName, PatternBenefit benefit, MLIRContext *ctx)
      : ConversionPattern(rootName, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    LLVM_DEBUG(llvm::outs() << "XTenBinaryOpConversion:\n");
    LLVM_DEBUG(op->getBlock()->print(llvm::outs()));

    rewriter.setInsertionPointAfter(op);
    Torch::BaseTensorType tensorType
      = op->getOperand(0).getType().cast<Torch::BaseTensorType>();

    auto sizes = tensorType.getSizes();

    MemRefType memrefTy = MemRefType::get(sizes,
                                                tensorType.getDtype(),
                                                {}, 0);
    Value result = rewriter.create<memref::AllocOp>(loc, memrefTy);
    Value argA = xten::MemRefTypeCast(rewriter, operands[0]);
    Value argB = xten::MemRefTypeCast(rewriter, operands[1]);

    SmallVector<int64_t, 4> lbs(sizes.size(), 0);
    SmallVector<int64_t, 4> steps(sizes.size(), 1);

    buildAffineLoopNest(
      rewriter, loc, lbs, sizes, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        SmallVector<Value, 4> indices;
        for (int i=0, e=sizes.size(); i<e; i++)
          indices.push_back(ivs[i]);
        auto ident = AffineMap::getMultiDimIdentityMap(sizes.size(),
                                                       op->getContext());
        auto loadA = rewriter.create<AffineLoadOp>(loc, argA, ident, indices);
        auto loadB = rewriter.create<AffineLoadOp>(loc, argB, ident, indices);
        auto binop = static_cast<const T*>(this)->emitBinaryOp(op, tensorType, rewriter, loadA, loadB);
        rewriter.create<AffineStoreOp>(loc, binop, result, ident, indices);
      });

    for (auto it = Block::iterator(op),ie=rewriter.getInsertionPoint(); it!=ie; ++it) {
       if (auto afo = dyn_cast<AffineForOp>(it))
        afo->setAttr("affine_opt_label", StringAttr::get(op->getContext(), "xten.binary_op"));
    }

    auto tensor_result =
        xten::TensorTypeCast(rewriter, result, op->getResult(0).getType());
    rewriter.replaceOp(op, {tensor_result});
    return success();
  }
};

class XTenMulOpConversion : public XTenBinaryOpConversion<XTenMulOpConversion> {
public:
  explicit XTenMulOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(xten::MulOp::getOperationName(), 1, context) {}

  Value
  emitBinaryOp(Operation *op, Torch::BaseTensorType tensorResultTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (FloatType::getF32(op->getContext()) == tensorResultTy.getDtype())
      return rewriter.create<mlir::arith::MulIOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::MulIOp>(op->getLoc(), a, b);
  }
};

class XTenAddOpConversion : public XTenBinaryOpConversion<XTenAddOpConversion> {
public:
  explicit XTenAddOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(xten::AddOp::getOperationName(), 1, context) {}

  Value
  emitBinaryOp(Operation *op, Torch::BaseTensorType tensorResultTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (FloatType::getF32(op->getContext()) == tensorResultTy.getDtype())
      return rewriter.create<mlir::arith::AddFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::AddFOp>(op->getLoc(), a, b);
  }
};


class XTenToAffinePass : public xilinx::xten::XTenToAffineBase<XTenToAffinePass> {

public:
  XTenToAffinePass() = default;
  XTenToAffinePass(const XTenToAffinePass &pass){};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<AffineDialect>();
     registry.insert<memref::MemRefDialect>();
     registry.insert<Torch::TorchDialect,
                     TorchConversion::TorchConversionDialect>();
  }

  ListOption<unsigned> clLoopOrder{*this, "xten-loop-order",
                                        llvm::cl::desc("XTen loop ordering applied in operator loweing to affine loops"),
                                        llvm::cl::ZeroOrMore,
                                        llvm::cl::CommaSeparated};


  // Initialize loop order from the command line or the default ordering
  void initLoopOrder();

  // Default size of loop ordering array 
  constexpr static unsigned kDefaultLoopOrderSize = 5;

  void runOnOperation() override {

    initLoopOrder();
    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    RewritePatternSet patterns(&getContext());
//    populateWithGenerated(context, patterns);

    patterns.insert<XTenAddConstantOpConversion,
                    XTenAddOpConversion,
                    XTenMulOpConversion>(context);

    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);

    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           memref::MemRefDialect,
                           StandardOpsDialect, scf::SCFDialect,
                           TorchConversion::TorchConversionDialect>();
    // target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    //    return typeConverter.isSignatureLegal(op.getType());
    // });

    target.addLegalOp<xten::Conv2dOp>();

    target.addDynamicallyLegalOp<Torch::AtenConv2dOp>([&](Torch::AtenConv2dOp conv2d) {
        Value weight = conv2d.weight();
        ShapedType weightTy = weight.getType().cast<ShapedType>();
        uint64_t kernel_h = weightTy.getDimSize(2);
        uint64_t kernel_w = weightTy.getDimSize(3);
        if (kernel_h == 1 && kernel_w == 1) {
          return false;
        }
        else if (kernel_h == 3 && kernel_w == 3) {
          return false;
        }
        return true;
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering XTen\n");
      signalPassFailure();
      //assert(0);
    }
  
    module.walk([&](Operation *op) {
      if (dyn_cast<xten::Conv2dOp>(op)) {
        (void)lowerConv2d(op);
      }
    });
  }

private:

};


/// Initializes loop order option parameters from the command line. If the
/// command line parameters are not set, use the default loop order
void XTenToAffinePass::initLoopOrder() {
  xten::Conv2dLoopOrder.clear();
  if (clLoopOrder.size() == kDefaultLoopOrderSize) {
    for (unsigned i = 0; i < clLoopOrder.size(); ++i) {
      xten::Conv2dLoopOrder.push_back(clLoopOrder[i]);
      LLVM_DEBUG(llvm::outs() << "clLoopOrder[" << i << "] = " << 
                  clLoopOrder[i] << "\n");
    }
  }
  else {
    for (unsigned i = 0; i < kDefaultLoopOrderSize; ++i) {
      xten::Conv2dLoopOrder.push_back(i);
      LLVM_DEBUG(llvm::outs() << "clLoopOrder[" << i << "] = " << 
                  i << "\n");
    }
  }
}


} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<Pass> createXTenToAffinePass() {
  return std::make_unique<XTenToAffinePass>();
}

} // namespace xten
} // namespace xilinx
