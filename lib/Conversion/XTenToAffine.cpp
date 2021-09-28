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
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"

#include "xten/Conversion/XTenToAffinePass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

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

using constInt = edsc::intrinsics::std_constant_int;
using constFloat = edsc::intrinsics::std_constant_float;
using constIndex = edsc::intrinsics::std_constant_index;

namespace xilinx {
namespace xten {

std::vector<uint64_t> Conv2dLoopOrder;
std::vector<uint64_t> Conv2dTileSizes;

}
}

namespace {

/// Create a type cast to memref
Value MemRefTypeCast(OpBuilder &builder, Value val) {
  if (val.getType().isa<MemRefType>())
    return val;
  auto tensorTy = val.getType().dyn_cast<TensorType>();
  if (!tensorTy)
    return val;
  auto memRefType = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), {}, 0);
  return builder.create<memref::BufferCastOp>(val.getLoc(), memRefType, val).getResult();
}

/// Create a type cast to tensor
Value TensorTypeCast(OpBuilder &builder, Value val) {
  if (val.getType().isa<TensorType>())
    return val;
  auto refType = val.getType().dyn_cast<MemRefType>();
  if (!refType)
    return val;
  return builder.create<memref::TensorLoadOp>(val.getLoc(), val).getResult();
}

void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
  if (auto co = op.getDefiningOp<NPCOMP::aten::ConstantOp>()) {
    DenseElementsAttr a = co->template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a.getIntValues())
      v.push_back(i.getSExtValue());
  }
  else if (auto co = op.getDefiningOp<NPCOMP::Basicpy::BuildListOp>()) {
    for (auto o : op.getDefiningOp()->getOperands())
      v.push_back(o.template getDefiningOp<ConstantIntOp>().getValue());
  }
}

LogicalResult
lowerConv2d(Operation *op) {

  auto conv2d = cast<xten::Conv2dOp>(op);
  auto loc = conv2d.getLoc();
  //auto module = op->getParentOfType<ModuleOp>();
  OpBuilder builder(op);
  edsc::ScopedContext scope(builder, loc);

  Value zero = constIndex(0);

  std::vector<Operation*> erasedOps;

  TensorType resultTy = conv2d.getResult().getType().cast<TensorType>();
  MemRefType memRefResultTy = MemRefType::get(resultTy.getShape(),
                                              resultTy.getElementType(),
                                              {}, 0);

  auto result = builder.create<memref::AllocOp>(loc, memRefResultTy);
  Value input = MemRefTypeCast(builder, conv2d.input());
  Value weight = MemRefTypeCast(builder, conv2d.weight());
  Value bias = MemRefTypeCast(builder, conv2d.bias());

  // get layer name to add to loop nest labels 
  std::string layer_name; 
  if (!op->getAttrs().empty()) {
      for (NamedAttribute attr : op->getAttrs()) {
          std::string attr_key = attr.first.c_str();
          if (attr_key.compare("name") == 0) {
              llvm::raw_string_ostream output(layer_name);
              attr.second.print(output);
              layer_name.pop_back();
              layer_name.erase(layer_name.begin());
          }
          layer_name.append(".");
      }
  }


  std::vector<int64_t> stride, padding, dilation;
  unpack_int_list(conv2d.stride(), stride);
  unpack_int_list(conv2d.padding(), padding);
  unpack_int_list(conv2d.dilation(), dilation);

  auto ctx = op->getContext();

  edsc::MemRefBoundsCapture vOutput(result);
  edsc::MemRefBoundsCapture vInput(input);
  edsc::MemRefBoundsCapture vWeight(weight);

  edsc::intrinsics::MemRefIndexedValue iOutput(result);
  edsc::intrinsics::MemRefIndexedValue iInput(input);
  edsc::intrinsics::MemRefIndexedValue iWeight(weight);

  if (bias.getType().isa<ShapedType>()) {
    edsc::intrinsics::MemRefIndexedValue iBias(bias);
    auto loop_body = [&](ValueRange ivs) {
      Value ofm_batch = ivs[0], ofm_channel = ivs[1];
      Value ofm_row = ivs[2], ofm_col = ivs[3];
      iOutput(ofm_batch, ofm_channel, ofm_row, ofm_col) = iBias(ofm_channel);
    };
    edsc::affineLoopNestBuilder({zero, zero, zero, zero},
                                {vOutput.ub(0), vOutput.ub(1), vOutput.ub(2), vOutput.ub(3)},
                                {1,1,1,1}, loop_body);
    auto a = cast<AffineForOp>(--builder.getInsertionPoint());
    a->setAttr("affine_opt_label", StringAttr::get(op->getContext(), layer_name + "xten.conv2d_op_bias"));
  }

  //
  // Construct a conv2d loop nest
  //
  Operation *conv_mul = nullptr;
  Operation *conv_add = nullptr;

  uint64_t batch_hw = 4;
  uint64_t ifm_channels_hw = 32;
  uint64_t ofm_channels_hw = 32;
  if (xten::Conv2dTileSizes.size()) {
    batch_hw = xten::Conv2dTileSizes[0];
    ofm_channels_hw = xten::Conv2dTileSizes[1];
    ifm_channels_hw = xten::Conv2dTileSizes[2];
  }

  FuncOp hwPlaceholderOp;

  // batch=0, ofm=1, ifm=2
  std::vector<uint64_t> loopOrder{0,1,2,3,4};
  if (xten::Conv2dLoopOrder.size()) {
    loopOrder = xten::Conv2dLoopOrder;
  }

  auto body_builder = [&](ValueRange ivs) {
    Value ofm_batch = ivs[ 1 ], ofm_channel = ivs[ 2 ], ifm_channel = ivs[ 3 ];
    Value ofm_row = ivs[0], ofm_col = ivs[4];

    std::vector<Type> retTy;
    std::vector<Value> ofm_indices{ofm_batch, ofm_channel, ofm_row, ofm_col};
    auto ident_4d = AffineMap::getMultiDimIdentityMap(4, ctx);
    edsc::affineLoopNestBuilder(
    {zero, zero},
    {vWeight.ub(2), vWeight.ub(3)},
    {1,1}, [&](ValueRange ivs3) {
      Value kx = ivs3[0], ky = ivs3[1];
      // ifm_row = ofm_row * stride + (ky - pad)
      // ifm_col = ofm_col * stride + (kx - pad);
      auto ifm_row_col_expr = getAffineDimExpr(0, ctx) *
        getAffineConstantExpr(stride[0], ctx) +
        (getAffineDimExpr(1, ctx) - getAffineConstantExpr(padding[0], ctx));
      auto ifm_row_col_map = AffineMap::get(2, 0, ifm_row_col_expr);
      std::vector<Value> ifm_row_ops{ofm_row, ky};
      std::vector<Value> ifm_col_ops{ofm_col, kx};
      auto b = edsc::ScopedContext::getBuilderRef();
      auto ifm_row = b.create<AffineApplyOp>(loc, ifm_row_col_map, ifm_row_ops);
      auto ifm_col = b.create<AffineApplyOp>(loc, ifm_row_col_map, ifm_col_ops);

      std::vector<Value> w_indices{ofm_channel, ifm_channel, ky, kx};
      std::vector<Value> ifm_indices{ofm_batch, ifm_channel, ifm_row, ifm_col};

      auto partial_load = b.create<AffineLoadOp>(loc, result, ident_4d, ofm_indices);
      auto input_load = b.create<AffineLoadOp>(loc, input, ident_4d, ifm_indices);
      auto weight_load = b.create<AffineLoadOp>(loc, weight, ident_4d, w_indices);
      conv_mul = b.create<MulFOp>(loc, input_load, weight_load);
      conv_add = b.create<AddFOp>(loc, conv_mul->getResult(0), partial_load);
      /*auto partial_store =*/ b.create<AffineStoreOp>(loc, conv_add->getResult(0), result, ident_4d, ofm_indices);
    });
  };

  // upper bounds
  std::vector<Value> ubs;
  for (auto i : loopOrder) {
    std::vector<Value> bounds{vOutput.ub(0), vOutput.ub(1), vInput.ub(1), vOutput.ub(2), vOutput.ub(3)};
    ubs.push_back(bounds[i]);
  }
  edsc::affineLoopNestBuilder({zero, zero, zero, zero, zero},
                              ubs,
                              {1,1,1,1,1},
                              body_builder);

  auto afo = cast<AffineForOp>(--builder.getInsertionPoint());
  afo->setAttr("affine_opt_label", StringAttr::get(op->getContext(), layer_name + "xten.conv2d_op"));

  LLVM_DEBUG(llvm::outs() << "\nInitial Conv2d loop nest:\n");
  LLVM_DEBUG(op->getBlock()->print(llvm::outs()));
  auto tensor_result = TensorTypeCast(builder,result);
  op->getResult(0).replaceAllUsesWith(tensor_result);
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
    auto mmult = cast<xten::MMOp>(op);
    auto loc = mmult.getLoc();

    using namespace edsc;
    edsc::ScopedContext scope(rewriter, loc);

    rewriter.setInsertionPointAfter(op);

    Type resultTy = mmult.getResult().getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    //Value result = rewriter.create<aten::AcapAllocOp>(loc, memRefResultTy, ArrayRef<Value>{});
    Value result = rewriter.create<memref::AllocOp>(loc, memRefResultTy);
    Value lhs = MemRefTypeCast(rewriter, operands[0]);
    Value rhs = MemRefTypeCast(rewriter, operands[1]);
    Value zero = constIndex(0);

    edsc::MemRefBoundsCapture vRes(result), vLHS(lhs), vRHS(rhs);
    edsc::intrinsics::MemRefIndexedValue iRes(result), iLHS(lhs), iRHS(rhs);

    if (vRes.rank() != 2) {
      return failure();
    }

    Value i, j;
    Value M(vRes.ub(0));
    Value N(vRes.ub(1));

    affineLoopNestBuilder({zero, zero}, {M, N},
                          {1,1}, [&] (ValueRange ivs) {
                            Value i = ivs[0]; Value j = ivs[1];
                            std::vector<Value> indices{i, j};
                            auto ident_2d = AffineMap::getMultiDimIdentityMap(2, op->getContext());
                            auto load0 = rewriter.create<AffineLoadOp>(loc, lhs, ident_2d, indices);
                            auto load1 = rewriter.create<AffineLoadOp>(loc, rhs, ident_2d, indices);
                            auto add = rewriter.create<MulFOp>(loc, load0, load1);
                            /*auto store =*/ rewriter.create<AffineStoreOp>(loc, result, add, ident_2d, indices);
                          });

    auto tensor_result = TensorTypeCast(rewriter,result);
    rewriter.replaceOp(op, {tensor_result});
    return success();
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

    edsc::ScopedContext scope(rewriter, loc);

    rewriter.setInsertionPointAfter(op);

    Type resultTy = addOp.getResult().getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    LLVM_DEBUG(llvm::outs() << "\nIn XTenAddConstantOpConversion\n");
    LLVM_DEBUG(op->getBlock()->print(llvm::outs()));

    Value result = rewriter.create<memref::AllocOp>(loc, memRefResultTy);
    Value lhs = MemRefTypeCast(rewriter, operands[0]);
    auto rhs = cast<mlir::ConstantOp>(operands[1].getDefiningOp()).getValue();

    auto f32Type = FloatType::getF32(op->getContext());
    auto i32Type = IntegerType::get(op->getContext(),32);
    bool isFloatOp = (f32Type == tensorResultTy.getElementType()); 

    SmallVector<int64_t, 4> lbs(tensorResultTy.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensorResultTy.getRank(), 1);

    buildAffineLoopNest(
      rewriter, loc, lbs, tensorResultTy.getShape(), steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        SmallVector<Value, 4> indices;
        auto ident = AffineMap::getMultiDimIdentityMap(ivs.size(),
                                                       builder.getContext());
        auto load = builder.create<AffineLoadOp>(loc, lhs, ident, ivs);
        std::vector<uint64_t> index{};
        Value add = nullptr;
        if (isFloatOp) {
          auto add_const = rhs.cast<DenseElementsAttr>().getValue<llvm::APFloat>(index);
          add = builder.create<AddFOp>(loc, load, constFloat(add_const, f32Type));
        }
        else {
          auto add_const = rhs.cast<DenseElementsAttr>().getValue<int>(index);
          add = builder.create<AddIOp>(loc, load, constInt(add_const, i32Type));
        }
        builder.create<AffineStoreOp>(loc, add, result, ident, ivs);
      });

    for (auto it = Block::iterator(op),ie=rewriter.getInsertionPoint(); it!=ie; ++it) {
       if (auto afo = dyn_cast<AffineForOp>(it))
        afo->setAttr("affine_opt_label", StringAttr::get(op->getContext(), "affine_opt"));
    }

    auto tensor_result = TensorTypeCast(rewriter,result);
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

    edsc::ScopedContext scope(rewriter, loc);

    rewriter.setInsertionPointAfter(op);
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);
    Value result = rewriter.create<memref::AllocOp>(loc, memRefResultTy);
    Value argA = MemRefTypeCast(rewriter, operands[0]);
    Value argB = MemRefTypeCast(rewriter, operands[1]);

    Value zero = constIndex(0);
    SmallVector<Value, 4> lbs(tensorResultTy.getRank(), zero);
    SmallVector<Value, 4> ubs;
    SmallVector<int64_t, 4> steps(tensorResultTy.getRank(), 1);

    edsc::MemRefBoundsCapture vRes(result);
    for (int i=0, e=tensorResultTy.getRank(); i<e; i++)
      ubs.push_back(vRes.ub(i));

    edsc::affineLoopNestBuilder(lbs, ubs, steps,
      [&](ValueRange ivs) {
        SmallVector<Value, 4> indices;
        for (int i=0, e=tensorResultTy.getRank(); i<e; i++)
          indices.push_back(ivs[i]);
        auto ident = AffineMap::getMultiDimIdentityMap(tensorResultTy.getRank(),
                                                       op->getContext());
        auto loadA = rewriter.create<AffineLoadOp>(loc, argA, ident, indices);
        auto loadB = rewriter.create<AffineLoadOp>(loc, argB, ident, indices);
        auto binop = static_cast<const T*>(this)->emitBinaryOp(op, tensorResultTy, rewriter, loadA, loadB);
        rewriter.create<AffineStoreOp>(loc, binop, result, ident, indices);
      });

    for (auto it = Block::iterator(op),ie=rewriter.getInsertionPoint(); it!=ie; ++it) {
       if (auto afo = dyn_cast<AffineForOp>(it))
        afo->setAttr("affine_opt_label", StringAttr::get(op->getContext(), "xten.binary_op"));
    }

    auto tensor_result = TensorTypeCast(rewriter,result);
    rewriter.replaceOp(op, {tensor_result});
    return success();

  }
};

class XTenMulOpConversion : public XTenBinaryOpConversion<XTenMulOpConversion> {
public:
  explicit XTenMulOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(xten::MulOp::getOperationName(), 1, context) {}

  Value
  emitBinaryOp(Operation *op, TensorType tensorResultTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (FloatType::getF32(op->getContext()) == tensorResultTy.getElementType())
      return rewriter.create<MulFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<MulIOp>(op->getLoc(), a, b);
  }
};

class XTenAddOpConversion : public XTenBinaryOpConversion<XTenAddOpConversion> {
public:
  explicit XTenAddOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(xten::AddOp::getOperationName(), 1, context) {}

  Value
  emitBinaryOp(Operation *op, TensorType tensorResultTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (FloatType::getF32(op->getContext()) == tensorResultTy.getElementType())
      return rewriter.create<AddFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<AddIOp>(op->getLoc(), a, b);
  }
};


class XTenToAffinePass : public PassWrapper<XTenToAffinePass,
                                           OperationPass<ModuleOp>> {

public:
  XTenToAffinePass() = default;
  XTenToAffinePass(const XTenToAffinePass &pass){};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<AffineDialect>();
     registry.insert<memref::MemRefDialect>();
     registry.insert<NPCOMP::aten::ATenDialect>();
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
    OwningRewritePatternList patterns(&getContext());
//    populateWithGenerated(context, patterns);

    patterns.insert<XTenAddConstantOpConversion,
                    XTenAddOpConversion,
                    XTenMulOpConversion>(context);

    populateFuncOpTypeConversionPattern(patterns,
                                        typeConverter);

    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           memref::MemRefDialect,
                           StandardOpsDialect, scf::SCFDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
       return typeConverter.isSignatureLegal(op.getType());
    });

    target.addLegalOp<xten::Conv2dOp>();

    target.addDynamicallyLegalOp<NPCOMP::aten::ConvolutionOp>([&](NPCOMP::aten::ConvolutionOp conv2d) {
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
