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

#include "xten/Conversion/XTenToLinalgPass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "xten-to-linalg-pass"

using namespace mlir;
using namespace xilinx::xten;
using namespace mlir::torch;

namespace {

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

    auto tensorTy = operands[0].getType().cast<Torch::BaseTensorType>();
    auto elementTy = tensorTy.getDtype();
    auto sizes = tensorTy.getSizes();
    auto rank = sizes.size();
    auto memRefResultTy = mlir::MemRefType::get(sizes, elementTy, {}, 0);

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

    auto tensor_cast =
        TensorTypeCast(rewriter, C->getResult(0), op->getResult(0).getType());
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
      return rewriter.create<mlir::arith::AddFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::AddIOp>(op->getLoc(), a, b);
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
      return rewriter.create<mlir::arith::MulFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::MulIOp>(op->getLoc(), a, b);
  }
};

class XTenMMOpConversion : public ConversionPattern {
public:
  explicit XTenMMOpConversion(MLIRContext *context)
      : ConversionPattern(MMOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<MMOp>(op);
    auto loc = mmult.getLoc();

    auto resultTy = op->getResult(0).getType();
    auto tTy = resultTy.cast<Torch::BaseTensorType>();
    auto oper0Ty = operands[0].getType().cast<Torch::BaseTensorType>();
    auto oper1Ty = operands[1].getType().cast<Torch::BaseTensorType>();
    auto dtype = tTy.getDtype();
    std::vector<int64_t> sizes{oper0Ty.getSizes()[1], oper1Ty.getSizes()[0]};
    auto memRefTy = mlir::MemRefType::get(sizes, dtype, {}, 0);

    auto A = MemRefTypeCast(rewriter, operands[0]);
    auto B = MemRefTypeCast(rewriter, operands[1]);
    auto C = rewriter.create<memref::AllocOp>(loc, memRefTy);
    rewriter
        .create<linalg::MatmulOp>(loc, TypeRange{}, ValueRange{A, B},
                                  ValueRange{C})
        .getResult(0);

    auto tensor_cast =
        TensorTypeCast(rewriter, C->getResult(0), op->getResult(0).getType());
    rewriter.replaceOp(op, tensor_cast);

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
    auto tensorTy = resultTy.cast<Torch::BaseTensorType>();
    auto memRefResultTy =
        mlir::MemRefType::get(tensorTy.getSizes(), tensorTy.getDtype(), {}, 0);

    auto C = rewriter.create<memref::AllocOp>(loc, memRefResultTy);

    rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B}, ValueRange{C});

    auto tensor_cast =
        TensorTypeCast(rewriter, C->getResult(0), op->getResult(0).getType());
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
    auto tensorResultTy = resultTy.cast<Torch::BaseTensorType>();
    auto memRefResultTy = mlir::MemRefType::get(
        tensorResultTy.getSizes(), tensorResultTy.getDtype(), {}, 0);

    Value C;
    if(mmult.PartialIn()) {
      C = mmult.PartialIn();
    } else {
      C = rewriter.create<memref::AllocOp>(loc, memRefResultTy).getResult();
    }

    rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B}, ValueRange{C});

    auto tensor_cast = TensorTypeCast(rewriter, C, op->getResult(0).getType());

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

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
     registry.insert<memref::MemRefDialect>();
     registry.insert<linalg::LinalgDialect>();
     registry.insert<Torch::TorchDialect,
                     TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    OwningRewritePatternList patterns(context);

    patterns.insert<XTenAddOpConversion,
                    XTenMulOpConversion,
                    XTenMMOpConversion,
                    XTenConv2dOpConversion,
                    XTenPartialConv2dReLUOpConversion>(context);

    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, StandardOpsDialect,
                           scf::SCFDialect, Torch::TorchDialect,
                           TorchConversion::TorchConversionDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering XTen to Linalg\n");
      signalPassFailure();
    }
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
