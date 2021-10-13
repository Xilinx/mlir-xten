//===- XTenToLinalg.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "xten/Conversion/XTenToLinalgPass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"

#define DEBUG_TYPE "xten-to-linalg-pass"

using namespace mlir;
using namespace xilinx::xten;

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

template <class T>
class XTenBinaryOpConversion : public ConversionPattern {
public:
  XTenBinaryOpConversion(StringRef rootName, PatternBenefit benefit, MLIRContext *ctx)
      : ConversionPattern(rootName, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto A = MemRefTypeCast(rewriter, operands[0]);
    auto B = MemRefTypeCast(rewriter, operands[1]);

    auto resultTy = op->getResult(0).getType();
    auto tensorResultTy = resultTy.cast<TensorType>();
    auto elementTy = tensorResultTy.getElementType();
    auto rank = tensorResultTy.getRank();
    auto memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto C = rewriter.create<memref::AllocOp>(loc, memRefResultTy);

    SmallVector<Value, 2> inputTensors{A,B};
    SmallVector<Value, 1> outputTensors{C};

    auto identMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap, 4> indexMap(3, identMap);

    /*auto linalgOp =*/ rewriter.create<linalg::GenericOp>(
      loc, ArrayRef<Type>(), inputTensors, outputTensors, indexMap,
      SmallVector<StringRef>(rank, getParallelIteratorTypeName()),
      "", static_cast<const T*>(this)->getDefaultLibraryFunc(),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result
          = static_cast<const T*>(this)->emitBinaryOp(op,
                                                      elementTy,
                                                      rewriter,
                                                      blockArgs[0],
                                                      blockArgs[1]);
        nestedBuilder.create<linalg::YieldOp>(loc, result);
      });

    auto tensor_cast = TensorTypeCast(rewriter, C->getResult(0));
    rewriter.replaceOp(op, tensor_cast);
    return success();
  }
};

class XTenAddOpConversion : public XTenBinaryOpConversion<XTenAddOpConversion> {
public:
  explicit XTenAddOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(AddOp::getOperationName(), 1, context) {}

  StringRef getDefaultLibraryFunc() const {
      return "xten_add_op";
  }

  Value
  emitBinaryOp(Operation *op, Type elementTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (elementTy.isa<FloatType>())
      return rewriter.create<AddFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<AddIOp>(op->getLoc(), a, b);
  }
};

class XTenMulOpConversion : public XTenBinaryOpConversion<XTenMulOpConversion> {
public:
  explicit XTenMulOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(MulOp::getOperationName(), 1, context) {}

  StringRef getDefaultLibraryFunc() const {
      return "xten_mul_op";
  }

  Value
  emitBinaryOp(Operation *op, Type elementTy,
               ConversionPatternRewriter &rewriter, Value a, Value b) const {
    if (elementTy.isa<FloatType>())
      return rewriter.create<MulFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<MulIOp>(op->getLoc(), a, b);
  }
};

class XTenMMOpConversion : public ConversionPattern {
public:
  explicit XTenMMOpConversion(MLIRContext *context, bool genTensors)
      : ConversionPattern(MMOp::getOperationName(), 1, context), generateTensors(genTensors) {}

  bool generateTensors;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<MMOp>(op);
    auto loc = mmult.getLoc();

    auto resultTy = op->getResult(0).getType();
    auto tensorTy = resultTy.cast<TensorType>();
    auto memRefTy = mlir::MemRefType::get(tensorTy.getShape(),
                                                tensorTy.getElementType(),
                                                {}, 0);

    if (generateTensors) {
      auto A = operands[0];
      auto B = operands[1];
      auto C = rewriter.create<linalg::InitTensorOp>(loc, tensorTy.getShape(), tensorTy.getElementType());
      rewriter.replaceOp(op,
        rewriter.create<linalg::MatmulOp>(loc,
                                          TypeRange{tensorTy},
                                          ValueRange{A, B},
                                          ValueRange{C}).getResult(0));
    } else {
      auto A = MemRefTypeCast(rewriter, operands[0]);
      auto B = MemRefTypeCast(rewriter, operands[1]);
      auto C = rewriter.create<memref::AllocOp>(loc, memRefTy);
      rewriter.create<linalg::MatmulOp>(loc,
                                        TypeRange{memRefTy},
                                        ValueRange{A, B},
                                        ValueRange{C}).getResult(0);

      auto tensor_cast = TensorTypeCast(rewriter, C->getResult(0));
      rewriter.replaceOp(op, tensor_cast);
    }
    return success();
  }
};

class XTenConv2dOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<Conv2dOp>(op);
    auto loc = mmult.getLoc();

    auto A = MemRefTypeCast(rewriter, operands[0]);
    auto B = MemRefTypeCast(rewriter, operands[1]);

    auto resultTy = op->getResult(0).getType();
    auto tensorResultTy = resultTy.cast<TensorType>();
    auto memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto C = rewriter.create<memref::AllocOp>(loc, memRefResultTy);

    rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B}, ValueRange{C});

    auto tensor_cast = TensorTypeCast(rewriter, C->getResult(0));
    rewriter.replaceOp(op, tensor_cast);
    return success();
  }
};

class XTenPartialConv2dReLUOpConversion : public ConversionPattern {
public:
  explicit XTenPartialConv2dReLUOpConversion(MLIRContext *context)
      : ConversionPattern(PartialConv2dReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<PartialConv2dReLUOp>(op);
    auto loc = mmult.getLoc();

    auto A = MemRefTypeCast(rewriter, operands[0]);
    auto B = MemRefTypeCast(rewriter, mmult.weight());

    auto resultTy = op->getResult(0).getType();
    auto tensorResultTy = resultTy.cast<TensorType>();
    auto memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    Value C;
    if(mmult.PartialIn()) {
      C = mmult.PartialIn();
    } else {
      C = rewriter.create<memref::AllocOp>(loc, memRefResultTy).getResult();
    }

    rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B}, ValueRange{C});

    auto tensor_cast = TensorTypeCast(rewriter, C);

    if(mmult.getNumResults() == 1)
      rewriter.replaceOp(op, tensor_cast);
    else
      rewriter.replaceOp(op, {tensor_cast, operands[0]});

    return success();
  }
};

class XTenToLinalgPass : public XTenToLinalgBase<XTenToLinalgPass> {

public:
  XTenToLinalgPass() = default;
  XTenToLinalgPass(const XTenToLinalgPass &pass) {};

  Option<bool> clLinalgOnTensors{
    *this, "linalg-on-tensors",
    llvm::cl::desc("Generate linalg operations on tensors instead of memrefs"),
    llvm::cl::init(false)
  };

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
     registry.insert<memref::MemRefDialect>();
     registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    OwningRewritePatternList patterns(context);

    patterns.insert<XTenMMOpConversion>(context, clLinalgOnTensors);

    patterns.insert<XTenAddOpConversion,
                    XTenMulOpConversion,
                    XTenConv2dOpConversion,
                    XTenPartialConv2dReLUOpConversion>(context);

    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, linalg::LinalgDialect,
                           memref::MemRefDialect,
                           StandardOpsDialect, scf::SCFDialect>();

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
       return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering XTen to Linalg\n");
      signalPassFailure();
      //assert(0);
    }

    // TODO: get rid of this out
    module.walk([&](linalg::MatmulOp op) {
      op->setAttr(
        linalg::LinalgTransforms::kLinalgTransformMarker,
        StringAttr::get(op->getContext(), "xten_mmult"));
    });
    module.walk([&](linalg::Conv2DNhwcHwcfOp op) {
      op->setAttr(
        linalg::LinalgTransforms::kLinalgTransformMarker,
        StringAttr::get( op->getContext(), "xten_conv2d"));
    });
  }

private:

};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<Pass> createXTenToLinalgPass() {
  return std::make_unique<XTenToLinalgPass>();
}

} // namespace xten
} // namespace xilinx
