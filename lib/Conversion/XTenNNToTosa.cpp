//===- XTenNNToTosa.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "xten/Conversion/XTenNNToTosaPass.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Convert the quantized integer type from signed to a signless version that
/// matches TOSA.
///
/// TOSA currently supports i8, i16, i32 tensors types. Here we convert the
/// arbitrary signed type from the XTenNN QDQ operators to one that can be
/// represented in TOSA.
/// @TODO: once TOSA moves from signless to signed integers with arbitrary
/// bit width we no longer need the type conversion.
///
///\param tensorType signed integer tensor type.
///\return TensorType new storage type for the \p tensorType.
TensorType getNewStorageType(TensorType tensorType) {
  assert(tensorType.getElementType().isSignlessInteger() &&
         "quantization should only work with integers");
  unsigned int integerBitwidth = tensorType.getElementTypeBitWidth();
  unsigned int storageBitWidth = 32;
  if (integerBitwidth <= 8) {
    storageBitWidth = 8;
  } else if (integerBitwidth <= 16) {
    storageBitWidth = 16;
  }
  return tensorType.cloneWith(
      {}, IntegerType::get(tensorType.getContext(), storageBitWidth));
}

RankedTensorType createSplatType(int64_t rank, Type elementType) {
  llvm::SmallVector<int64_t, 4> tmpShape;
  // On average the tensor rank will be four, if it is greater, use the
  // reserve function to ensure we do not reallocate upon each insertion if the
  // rank is greater.
  tmpShape.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    tmpShape.emplace_back(1);
  }
  return RankedTensorType::get(tmpShape, elementType);
}

/// Hold min and max values of an Integer type
struct IntegerMinMax {
  int64_t min = std::numeric_limits<int64_t>::min();
  int64_t max = std::numeric_limits<int64_t>::max();
};

/// Calculate the limits of the given Tensor Element Type.
///
///\param type the tensor type we are inspecting for the element type.
///\return IntegerMinMax for the element integer type of the tensor \p type. If
/// the element type is not a IntegerType then return the min and max of int64_t
IntegerMinMax calculateMinMaxOfElementType(TensorType type) {
  IntegerType intType = dyn_cast<IntegerType>(type.getElementType());
  if (!intType) {
    return IntegerMinMax{};
  }
  auto minValue =
      llvm::APSInt::getMinValue(intType.getWidth(), type.isUnsignedInteger());
  auto maxValue =
      llvm::APSInt::getMaxValue(intType.getWidth(), type.isUnsignedInteger());
  return IntegerMinMax{minValue.getSExtValue(), maxValue.getSExtValue()};
}

class QuantizeOp : public OpRewritePattern<amd::xten_nn::QuantizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amd::xten_nn::QuantizeOp quantizeOp,
                                PatternRewriter &rewriter) const override {
    // The QDQ operations only work on tensors, if they are not, then the
    // verifiers should find the error. At the moment, only signless tensors
    // are supported.
    auto outputType = cast<TensorType>(quantizeOp->getResult(0).getType());
    if (!outputType.getElementType().isSignlessInteger()) {
      return rewriter.notifyMatchFailure(
          quantizeOp.getLoc(),
          "only signless integer tensor types are supported.");
    }
    auto inputType = dyn_cast<TensorType>(quantizeOp->getOperand(0).getType());

    // Calculate (1 / 2 ^ shift)
    llvm::APFloat scale(std::pow(static_cast<float>(2.0),
                                 static_cast<float>(-quantizeOp.getShift())));

    // Create a constant that represents the (1 / 2 ^ shift)
    RankedTensorType constType =
        createSplatType(inputType.getRank(), rewriter.getF32Type());
    auto constOp = rewriter.create<tosa::ConstOp>(
        quantizeOp->getLoc(), constType,
        DenseFPElementsAttr::get(constType, {scale}));

    // Calculate (x / 2 ^ shift)
    auto mulOp = rewriter.create<tosa::MulOp>(
        quantizeOp.getLoc(), inputType, quantizeOp->getOperand(0),
        constOp->getResult(0), rewriter.getI8IntegerAttr(0));

    // TOSA only supports signed integers of i8, i16 or i32 here we convert our
    // si<?> to this types and add a clamp to mimic arbitrary bit width.
    TensorType newIntegerStorageType = getNewStorageType(outputType);
    // Cast from fp32 -> i<Bitwidth> where bit width is the supported storage
    // bit width. Either i8, i16 or i32
    auto castOp = rewriter.create<tosa::CastOp>(
        quantizeOp->getLoc(), newIntegerStorageType, mulOp->getResult(0));

    // Find the max and min of the signed integer type.
    IntegerMinMax intLimits = calculateMinMaxOfElementType(outputType);

    // Clamp the integer to the min and max we calculated for this custom
    // bit width
    auto clampOp = rewriter.createOrFold<tosa::ClampOp>(
        quantizeOp->getLoc(), newIntegerStorageType, castOp->getResult(0),
        rewriter.getI64IntegerAttr(intLimits.min),
        rewriter.getI64IntegerAttr(intLimits.max),
        rewriter.getF32FloatAttr((float)intLimits.min),
        rewriter.getF32FloatAttr((float)intLimits.max));

    // Use an unrealized conversion cast to match the original output type.
    // We convert I back to SI because TOSA does not support the SI type
    // explicitly.
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        quantizeOp->getLoc(), quantizeOp->getResult(0).getType(), clampOp);

    // Replace the original op with the new decomposition
    rewriter.replaceOp(quantizeOp, {unrealizedCast.getResult(0)});

    return success();
  }
};

class DequantizeOp : public OpRewritePattern<amd::xten_nn::DequantizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amd::xten_nn::DequantizeOp dequantizeOp,
                                PatternRewriter &rewriter) const override {
    // The QDQ operations only work on tensors, if they are not, then the
    // verifiers should find the error. At the moment, only signless tensors
    // are supported.
    auto inputType = cast<TensorType>(dequantizeOp->getOperand(0).getType());
    if (!inputType.getElementType().isSignlessInteger()) {
      return rewriter.notifyMatchFailure(
          dequantizeOp.getLoc(),
          "only signless integer tensor types are supported.");
    }

    TensorType newIntegerStorageType = getNewStorageType(inputType);
    // We need to convert from si<> to i8, i16 or i32 depending on the input
    // type bit width.
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        dequantizeOp.getLoc(), newIntegerStorageType,
        dequantizeOp->getOperand(0));

    // We can then cast from i<8,16,32> -> fp32
    auto castOp = rewriter.create<tosa::CastOp>(
        dequantizeOp->getLoc(), dequantizeOp->getResult(0).getType(),
        unrealizedCast.getResult(0));

    // Calculate the (x * 2 ^ shift) for the dequantize part
    llvm::APFloat scale(std::pow(static_cast<float>(2.0),
                                 static_cast<float>(dequantizeOp.getShift())));

    // Create a constant to hold the floating point scale we just calculated
    auto constType =
        createSplatType(inputType.getRank(), rewriter.getF32Type());
    auto constOp = rewriter.create<tosa::ConstOp>(
        dequantizeOp->getLoc(), constType,
        DenseFPElementsAttr::get(constType, {scale}));

    // Replace the dequantize op with the new operations we just created.
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        dequantizeOp, dequantizeOp->getResult(0).getType(),
        castOp->getResult(0), constOp->getResult(0),
        rewriter.getI8IntegerAttr(0));
    return success();
  }
};

class XTenNNToTosaPass
    : public xilinx::xten::XTenNNToTosaBase<XTenNNToTosaPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    RewritePatternSet patterns(context);

    patterns.insert<QuantizeOp, DequantizeOp>(context);
    // We insert a clamp to enforce non-standard TOSA dataypes. E.g. i6 signed
    // integer range described with an i8 value. However, in the case we use i8
    // and clamp to values of i8 (i.e. si8) then the clamp can be optimized away
    // and the following canonicalization will check/do that.
    tosa::ClampOp::getCanonicalizationPatterns(patterns, context);

    FrozenRewritePatternSet frozenSetOfPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(module, frozenSetOfPatterns))) {
      emitError(module->getLoc())
          << "failed to convert XTenNN quantization operations to TOSA.";
      signalPassFailure();
    }
  }
};

} // namespace

namespace amd {
namespace xten_nn {

std::unique_ptr<mlir::Pass> createXTenNNToTOSAPass() {
  return std::make_unique<XTenNNToTosaPass>();
}

} // namespace xten_nn
} // namespace amd
