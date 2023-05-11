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

/// Checks to see if the \p castOp is a dequantize operation.
///
///\param castOp check that the operation casts from i8, assuming it is signed
/// to f32.
/// \return true represents a dequantize cast
/// \return false does not represent a dequantize cast
bool isI8ToF32Cast(tosa::CastOp castOp) {
  TensorType inputTensorType =
      cast<TensorType>(castOp->getOperand(0).getType());
  TensorType outputTensorType =
      cast<TensorType>(castOp->getResult(0).getType());
  return outputTensorType.getElementType().isF32() &&
         inputTensorType.getElementType().isInteger(8) &&
         inputTensorType.getElementType().isSignlessInteger();
}

/// Checks to see if the \p castOp is a quantize operation.
///
///\param castOp check that the operation casts from f32
/// to i8, assuming it is signed.
///\return true represents a quantize cast
///\return false does not represent a quantize cast
bool isF32ToI8Cast(tosa::CastOp castOp) {
  TensorType inputTensorType =
      cast<TensorType>(castOp->getOperand(0).getType());
  TensorType outputTensorType =
      cast<TensorType>(castOp->getResult(0).getType());
  return inputTensorType.getElementType().isF32() &&
         outputTensorType.getElementType().isInteger(8) &&
         outputTensorType.getElementType().isSignlessInteger();
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

    // Dequantize is from i8 -> f32 here we need to check for that
    if (!isI8ToF32Cast(castOp)) {
      return rewriter.notifyMatchFailure(
          castOp->getLoc(), "dequantize operation input and output are not i8 "
                            "and f32 respectively.");
    }

    // Quantize is from f32 -> i8 here we need to check for that
    if (!isF32ToI8Cast(quantizeOp)) {
      return rewriter.notifyMatchFailure(castOp->getLoc(),
                                         "quantize operation input and output "
                                         "are not f32 and i8 respectively.");
    }

    auto outputTensorType =
        cast<TensorType>(quantizeOp->getOperand(0).getType());
    Type newOutputType = outputTensorType.cloneWith(
        {}, IntegerType::get(rewriter.getContext(), 8,
                             IntegerType::SignednessSemantics::Signed));

    auto newQuantizeOp = rewriter.create<amd::xten_nn::QuantizeOp>(
        quantizeOp->getLoc(), newOutputType, quantizeOp->getOperand(0),
        /*shift*/ (int32_t)0);
    rewriter.replaceOpWithNewOp<amd::xten_nn::DequantizeOp>(
        castOp, outputTensorType, newQuantizeOp->getResult(0),
        /*shift*/ (int32_t)0);
    rewriter.eraseOp(quantizeOp);

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

  LogicalResult matchAndRewrite(tosa::MulOp mulOp,
                                PatternRewriter &rewriter) const override {

    // The multiplication should have only a single constant value
    APFloat dequantizeScaleFactor(0.0);
    if (!m_ConstantFloat(&dequantizeScaleFactor)
             .match(mulOp.getOperand(1).getDefiningOp())) {
      return rewriter.notifyMatchFailure(mulOp.getOperand(1).getLoc(),
                                         "expected to be a constant.");
    }

    // We want to make sure we have QDQ operations surrounded by MULs.
    Operation *mulOperandOp = mulOp->getOperand(0).getDefiningOp();
    APFloat quantizeScaleFactor(0.0);
    auto isQDQPattern = m_Op<amd::xten_nn::DequantizeOp>(
        m_Op<amd::xten_nn::QuantizeOp>(m_Op<tosa::MulOp>(
            matchers::m_Any(), m_ConstantFloat(&quantizeScaleFactor))));
    if (mulOperandOp == nullptr || !isQDQPattern.match(mulOperandOp)) {
      return rewriter.notifyMatchFailure(mulOp->getLoc(),
                                         "expected mul->q->dq->mul pattern.");
    }
    auto dequantizeOp = cast<amd::xten_nn::DequantizeOp>(mulOperandOp);

    // The quantize operation will have it's reciprocal value constant folded.
    // So we need to calculate one over to get the scale factor back.
    APFloat recipValue((float)1.0);
    recipValue.divide(quantizeScaleFactor, APFloat::rmNearestTiesToEven);
    if (recipValue != dequantizeScaleFactor) {
      return rewriter.notifyMatchFailure(
          mulOp.getLoc(), "expected constants of both multiplications around "
                          "the QDQ to be equal.");
    }

    // Only power of two values are supported by the QDQ operations
    std::optional<int32_t> scaleFactor =
        getLog2Value(dequantizeScaleFactor.convertToFloat());
    if (!scaleFactor.has_value()) {
      return rewriter.notifyMatchFailure(
          mulOp.getLoc(), "constant is not a integer log2 value.");
    }

    auto quantizeOp = cast<amd::xten_nn::QuantizeOp>(
        dequantizeOp->getOperand(0).getDefiningOp());

    // Sum the shifts of the quantize, dequantize and update the operations
    llvm::APInt shiftSum(32, dequantizeOp.getShift(), true);
    bool overflow = false;
    llvm::APInt scaleFactorInt(32, scaleFactor.value(), true);
    shiftSum = shiftSum.sadd_ov(scaleFactorInt, overflow);
    if (overflow) {
      return rewriter.notifyMatchFailure(
          mulOp.getLoc(), "Adding the shifts of the mul and QDQ overflowed.");
    }
    quantizeOp.setShift((int32_t)shiftSum.getSExtValue());
    dequantizeOp.setShift((int32_t)shiftSum.getSExtValue());

    // Remove the multiplications around the QDQ
    auto *quantizeMulOp = quantizeOp->getOperand(0).getDefiningOp();
    quantizeOp->setOperand(0, quantizeMulOp->getOperand(0));
    rewriter.replaceOp(mulOp, dequantizeOp->getResults());
    rewriter.eraseOp(quantizeMulOp);

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
