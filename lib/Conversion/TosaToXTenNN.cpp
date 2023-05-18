//===- TosaToXTenNN.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/CommutativityUtils.h"
#include "xten/Conversion/TosaToXTenNNPass.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Checks if the first tensor is a F32Tensor and the second i8Tensor
///
///\param f32Tensor expected to be a F32Tensor
///\param i8Tensor expected to be a i8Tensor
///\return true if both arguments match their expected types.
///\return false otherwise
///
bool checkInputOutputType(TensorType f32Tensor, TensorType i8Tensor) {
  return f32Tensor.getElementType().isF32() &&
         i8Tensor.getElementType().isInteger(8) &&
         i8Tensor.getElementType().isSignlessInteger();
}

/// Get the Log2 \p value of the float. If the \p value is not an exact
/// power-of-two return none.
///
///\param value floating point value we are checking
///\return std::optional<int32_t> none if \p value is not a power of two and an
/// integer if it is.
std::optional<int32_t> getLog2Value(float value) {
  float log2Value = std::log2(value);
  float integerPart = 0.0;
  float fractionalPart = std::modf(log2Value, &integerPart);
  if (fractionalPart != 0.0) {
    return std::nullopt;
  }
  return (int32_t)integerPart;
}

/// Checks that operand(0) and the result(0) have the same type.
///
/// Used to check that broadcasting did not occur, for example, on
/// multiplication operations.
///
///\param operation we are inspecting. Assumes operand(0) and result(0) exist.
///\return true shape does not change from input to output
///\return false shape does change from input to output
bool sameInputAndOutputShape(mlir::Operation *operation) {
  assert(operation->getNumOperands() == 2 && operation->getNumResults() == 1 &&
         "expected operation with 2 inputs and one output.");
  return operation->getOperand(0).getType() ==
         operation->getResult(0).getType();
}

/// Checks to see that the two consecutive casts can be represented by quantize
/// and dequantize operations.
class CastsToQDQOps : public OpRewritePattern<tosa::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto quantizeOp =
        dyn_cast_or_null<tosa::CastOp>(castOp->getOperand(0).getDefiningOp());
    if (!quantizeOp) {
      return rewriter.notifyMatchFailure(castOp->getOperand(0).getLoc(),
                                         "expected cast -> cast pattern.");
    }

    if (!quantizeOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          quantizeOp.getLoc(),
          "expected the quantize operation to have a single use.");
    }

    TensorType inputTensorType = castOp.getInput().getType();
    TensorType outputTensorType = castOp.getOutput().getType();

    // Dequantize is from i8 -> f32 here we need to check for that
    if (!checkInputOutputType(/*f32*/ outputTensorType,
                              /*i8*/ inputTensorType)) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "dequantize operation input and output are not i8 "
                            "and f32 respectively.");
    }

    // Quantize is from f32 -> i8 here we need to check for that
    if (!checkInputOutputType(/*f32*/ quantizeOp.getInput().getType(),
                              /*i8*/ quantizeOp.getOutput().getType())) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
                                         "quantize operation input and output "
                                         "are not f32 and i8 respectively.");
    }

    Type newOutputType = outputTensorType.cloneWith(
        {}, IntegerType::get(rewriter.getContext(), 8,
                             IntegerType::SignednessSemantics::Signed));

    auto newQuantizeOp = rewriter.create<amd::xten_nn::QuantizeOp>(
        quantizeOp->getLoc(), newOutputType, quantizeOp->getOperand(0),
        /*shift*/ (int32_t)0);
    rewriter.replaceOpWithNewOp<amd::xten_nn::DequantizeOp>(
        castOp, outputTensorType, newQuantizeOp->getResult(0),
        /*shift*/ (int32_t)0);

    return success();
  }
};

/// Folds multiplications surrounding a QDQ pair into the operations if the
/// constants on the multiplications are power-of-two values and are equal.
/// We assume the first multiplication represents the quantize meaning 1/scale
/// and the second multiplication the dequantize or simply scale.
class FoldMulsToQDQOps : public OpRewritePattern<tosa::MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp dequantizeMulOp,
                                PatternRewriter &rewriter) const override {

    // We want to make sure we have QDQ operations surrounded by MULs.
    auto dequantizeOp = dequantizeMulOp->getOperand(0)
                            .getDefiningOp<amd::xten_nn::DequantizeOp>();
    APFloat quantizeScaleFactor(0.0);
    auto isQDQPattern = m_Op<amd::xten_nn::DequantizeOp>(
        m_Op<amd::xten_nn::QuantizeOp>(m_Op<tosa::MulOp>(
            matchers::m_Any(), m_ConstantFloat(&quantizeScaleFactor))));
    if (!dequantizeOp || !isQDQPattern.match(dequantizeOp)) {
      return rewriter.notifyMatchFailure(dequantizeMulOp->getLoc(),
                                         "expected mul->q->dq->mul pattern.");
    }

    // The multiplication should have only a single constant value
    APFloat dequantizeScaleFactor(0.0);
    if (!m_ConstantFloat(&dequantizeScaleFactor)
             .match(dequantizeMulOp.getOperand(1).getDefiningOp())) {
      return rewriter.notifyMatchFailure(dequantizeMulOp.getOperand(1).getLoc(),
                                         "expected to be a constant.");
    }

    auto quantizeOp = cast<amd::xten_nn::QuantizeOp>(
        dequantizeOp->getOperand(0).getDefiningOp());

    // Make sure these multiplications really only belong to the QDQ operations
    // and are used by no one else
    auto *quantizeMulOp = quantizeOp->getOperand(0).getDefiningOp();
    if (!quantizeMulOp->hasOneUse() || !dequantizeMulOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          dequantizeMulOp->getLoc(),
          "multiplications around the QDQ operations must have single user.");
    }

    if (!sameInputAndOutputShape(quantizeMulOp) ||
        !sameInputAndOutputShape(dequantizeMulOp)) {
      return rewriter.notifyMatchFailure(
          dequantizeMulOp.getLoc(),
          "i/o shape cannot change when multiplying due to broadcasting.");
    }

    // Attempt to convert the scale factors to a log2 base. And ensure that both
    // are equal. The quantization factor being the negation of the
    // dequantization factor
    std::optional<int32_t> quantizeLog2ScaleFactor =
        getLog2Value(quantizeScaleFactor.convertToFloat());
    std::optional<int32_t> dequantizeLog2ScaleFactor =
        getLog2Value(dequantizeScaleFactor.convertToFloat());
    if (!quantizeLog2ScaleFactor.has_value() ||
        !dequantizeLog2ScaleFactor.has_value() ||
        (quantizeLog2ScaleFactor.value() + dequantizeLog2ScaleFactor.value() !=
         0)) {
      return rewriter.notifyMatchFailure(
          dequantizeMulOp.getLoc(),
          "expected constants of both multiplications to be "
          "equal and power-of-two values.");
    }

    // Sum the shifts of the quantize, dequantize and update the operations
    llvm::APInt shiftSum(32, dequantizeOp.getShift(), true);
    bool overflow = false;
    llvm::APInt scaleFactorInt(32, dequantizeLog2ScaleFactor.value(), true);
    shiftSum = shiftSum.sadd_ov(scaleFactorInt, overflow);
    if (overflow) {
      return rewriter.notifyMatchFailure(
          dequantizeMulOp.getLoc(),
          "adding the shifts of the mul and QDQ overflowed.");
    }

    // Create new QDQ nodes with the updated shifts.
    auto newQOp = rewriter.create<amd::xten_nn::QuantizeOp>(
        quantizeOp->getLoc(), quantizeOp->getResult(0).getType(),
        quantizeMulOp->getOperand(0),
        /*shift*/ (int32_t)shiftSum.getSExtValue());
    auto newDQ = rewriter.create<amd::xten_nn::DequantizeOp>(
        dequantizeOp->getLoc(), dequantizeOp->getResult(0).getType(),
        newQOp->getResult(0), /*shift*/ (int32_t)shiftSum.getSExtValue());

    // Remove the multiplication around the quantize
    rewriter.replaceOp(dequantizeMulOp, newDQ->getResults());

    return success();
  }
};

class TosaToXTenNNPass
    : public xilinx::xten::TOSAToXTenNNBase<TosaToXTenNNPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    RewritePatternSet patterns(context);

    // Ensures constants on the add, mul, sub are on the RHS
    populateCommutativityUtilsPatterns(patterns);
    // Patterns for finding the QDQ and folding MULs.
    patterns.insert<CastsToQDQOps, FoldMulsToQDQOps>(context);

    FrozenRewritePatternSet frozenSetOfPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(module, frozenSetOfPatterns))) {
      emitError(module->getLoc())
          << "failed to convert TOSA to XTenNN Quantization "
             "Operations.";
      signalPassFailure();
    }
  }
};

} // namespace

namespace amd {
namespace xten_nn {

std::unique_ptr<Pass> createTOSAToXTenNNPass() {
  return std::make_unique<TosaToXTenNNPass>();
}

} // namespace xten_nn
} // namespace amd
