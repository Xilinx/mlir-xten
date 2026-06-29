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
/// tosa-mlir supports arbitrary bitwidth, but for historic reasons signless is
/// used instead of signed
///
///\param tensorType signed integer tensor type.
///\return TensorType new storage type for the \p tensorType.
TensorType getNewStorageType(TensorType tensorType) {
  assert(tensorType.getElementType().isInteger() &&
         "quantization should only work with integers");
  return tensorType.cloneWith(
      {}, IntegerType::get(tensorType.getContext(),
                           tensorType.getElementTypeBitWidth(),
                           (tensorType.getElementType().isUnsignedInteger())
                               ? IntegerType::Unsigned
                               : IntegerType::Signless));
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

namespace {
APFloat convertF32AttrToFloatTy(FloatAttr attr, Type typeToConvertTo) {
  // Convert from f32 to the float type that is actually used
  assert(attr.getType().isF32());
  assert(isa<FloatType>(typeToConvertTo));
  auto floatResultType = cast<FloatType>(typeToConvertTo);
  APFloat scale = attr.getValue();
  bool losesInfo;
  // Ignore inaccuracies, there is nothing we can do.
  [[maybe_unused]] const auto conversionResult =
      scale.convert(floatResultType.getFloatSemantics(),
                    llvm::RoundingMode::NearestTiesToEven, &losesInfo);
  return scale;
}

Value getZeroShift(PatternRewriter &rewriter, Location loc) {
  auto shiftValueType = RankedTensorType::get({1}, rewriter.getI8Type());
  auto shiftValueAttr = DenseElementsAttr::get(shiftValueType, {int8_t(0)});

  auto shiftValue =
      rewriter.create<tosa::ConstOp>(loc, shiftValueType, shiftValueAttr);
  return shiftValue;
}

Value createTosaI32Const(PatternRewriter &rewriter, Location loc,
                         ArrayRef<int32_t> values) {
  auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                    rewriter.getI32Type());
  return rewriter
      .create<tosa::ConstOp>(loc, type, DenseIntElementsAttr::get(type, values))
      .getResult();
}
} // namespace

class QuantizeOp : public OpRewritePattern<amd::xten_nn::QuantizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amd::xten_nn::QuantizeOp quantizeOp,
                                PatternRewriter &rewriter) const override {
    // The QDQ operations only work on tensors, if they are not, then the
    // verifiers should find the error.
    const auto outputType =
        cast<TensorType>(quantizeOp->getResult(0).getType());
    const auto inputType =
        cast<TensorType>(quantizeOp->getOperand(0).getType());
    const auto inputElementType = inputType.getElementType();

    // Convert the scale from f32 to the float type that is actually used
    const llvm::APFloat scale =
        convertF32AttrToFloatTy(quantizeOp.getScaleAttr(), inputElementType);
    const llvm::APFloat scaleReciprocal =
        llvm::APFloat::getOne(scale.getSemantics()) / scale;

    const RankedTensorType constType =
        createSplatType(inputType.getRank(), inputElementType);
    auto constOp = rewriter.create<tosa::ConstOp>(
        quantizeOp->getLoc(), constType,
        DenseFPElementsAttr::get(constType, {scaleReciprocal}));

    auto mulOp = rewriter.create<tosa::MulOp>(
        quantizeOp.getLoc(), inputType, quantizeOp->getOperand(0),
        constOp->getResult(0), getZeroShift(rewriter, quantizeOp.getLoc()));

    mlir::Value castFrom = mulOp->getResult(0);
    if (!quantizeOp.getZeroPoint().isZero()) {
      // Do the zero_point calculation on int32 to match the reference
      // implementation:
      // https://github.com/onnx/onnx/blob/3d5acaf3e23ae8db7ac01b8cfedb17b8817121f4/onnx/reference/ops/op_quantize_linear.py#L177
      const auto int32InputType =
          inputType.cloneWith({}, rewriter.getI32Type());
      auto castToint32Op = rewriter.create<tosa::CastOp>(
          quantizeOp->getLoc(), int32InputType, mulOp.getResult());

      const auto constAddType =
          createSplatType(inputType.getRank(), outputType.getElementType());
      auto zeroPointConstOp = rewriter.create<tosa::ConstOp>(
          quantizeOp.getLoc(), constAddType,
          DenseIntElementsAttr::get(constAddType, {quantizeOp.getZeroPoint()}));
      auto zeroPointCastOp = rewriter.create<tosa::CastOp>(
          quantizeOp.getLoc(), int32InputType, zeroPointConstOp.getResult());
      auto zeroPointAddOp = rewriter.create<tosa::AddOp>(
          quantizeOp.getLoc(), int32InputType, castToint32Op.getResult(),
          zeroPointCastOp.getResult());
      castFrom = zeroPointAddOp->getResult(0);
    }
    const TensorType newIntegerStorageType = getNewStorageType(outputType);
    auto castOp = rewriter.create<tosa::CastOp>(
        quantizeOp->getLoc(), newIntegerStorageType, castFrom);

    // Use an unrealized conversion cast to match the original output type.
    // We convert I back to SI because TOSA does not support the SI type
    // explicitly.
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        quantizeOp->getLoc(), quantizeOp->getResult(0).getType(),
        castOp->getResult(0));

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
    // verifiers should find the error.
    const auto resultElementType =
        dequantizeOp.getResult().getType().getElementType();
    const auto inputType =
        cast<TensorType>(dequantizeOp->getOperand(0).getType());

    TensorType newIntegerStorageType = getNewStorageType(inputType);
    // We need to convert from si<> to i8, i16 or i32 depending on the input
    // type bit width.
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        dequantizeOp.getLoc(), newIntegerStorageType,
        dequantizeOp->getOperand(0));

    auto castOp = rewriter.create<tosa::CastOp>(
        dequantizeOp->getLoc(), dequantizeOp->getResult(0).getType(),
        unrealizedCast.getResult(0));

    // Do the zero_point sub on the float type to to avoid underflows
    const auto constSubType =
        createSplatType(inputType.getRank(), inputType.getElementType());
    auto zeroPointConstOp = rewriter.create<tosa::ConstOp>(
        dequantizeOp.getLoc(), constSubType,
        DenseIntElementsAttr::get(constSubType, {dequantizeOp.getZeroPoint()}));
    auto constSubCastOp = rewriter.create<tosa::CastOp>(
        dequantizeOp.getLoc(), dequantizeOp.getResult().getType(),
        zeroPointConstOp.getResult());
    auto zeroPointSubOp = rewriter.create<tosa::SubOp>(
        dequantizeOp.getLoc(), dequantizeOp.getResult().getType(),
        castOp.getResult(), constSubCastOp.getResult());

    // Convert the scale from f32 to the float type that is actually used
    const llvm::APFloat scale =
        convertF32AttrToFloatTy(dequantizeOp.getScaleAttr(), resultElementType);

    // Create a constant to hold the floating point scale we just calculated
    auto constType = createSplatType(inputType.getRank(), resultElementType);
    auto constOp = rewriter.create<tosa::ConstOp>(
        dequantizeOp->getLoc(), constType,
        DenseFPElementsAttr::get(constType, {scale}));

    // Replace the dequantize op with the new operations we just created.
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        dequantizeOp, dequantizeOp->getResult(0).getType(),
        zeroPointSubOp->getResult(0), constOp->getResult(0),
        getZeroShift(rewriter, dequantizeOp.getLoc()));
    return success();
  }
};

class DepthToSpaceOp : public OpRewritePattern<amd::xten_nn::DepthToSpaceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amd::xten_nn::DepthToSpaceOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getX().getType());
    auto resultType = cast<RankedTensorType>(op.getY().getType());
    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "depth_to_space lowering requires static shapes");

    const int64_t blockSize = op.getBlocksize();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    const int64_t channels = inputShape[1];
    const int64_t blockSizeSquared = blockSize * blockSize;
    assert(blockSizeSquared > 0 && "blocksize must be positive");
    assert(channels % blockSizeSquared == 0 &&
           "input channel dimension must be divisible by blocksize squared");

    const int64_t outputChannels = channels / blockSizeSquared;
    SmallVector<int64_t> reshapeShape;
    SmallVector<int64_t> transposedShape{inputShape[0], outputChannels,
                                         inputShape[2], blockSize,
                                         inputShape[3], blockSize};
    SmallVector<int32_t> permutation;

    switch (op.getMode()) {
    case 1:
      // DCR: [N, BS, BS, C', H, W] -> [N, C', H, BS, W, BS].
      reshapeShape = {inputShape[0],  blockSize,     blockSize,
                      outputChannels, inputShape[2], inputShape[3]};
      permutation = {0, 3, 4, 1, 5, 2};
      break;
    case 2:
      // CRD: [N, C', BS, BS, H, W] -> [N, C', H, BS, W, BS].
      reshapeShape = {inputShape[0], outputChannels, blockSize,
                      blockSize,     inputShape[2],  inputShape[3]};
      permutation = {0, 1, 4, 2, 5, 3};
      break;
    default:
      return rewriter.notifyMatchFailure(op, "unsupported depth_to_space mode");
    }

    auto reshapedType =
        RankedTensorType::get(reshapeShape, inputType.getElementType());
    Value reshaped = rewriter
                         .create<tosa::ReshapeOp>(
                             op.getLoc(), reshapedType, op.getX(),
                             rewriter.getDenseI64ArrayAttr(reshapeShape))
                         .getResult();

    auto transposedType =
        RankedTensorType::get(transposedShape, inputType.getElementType());
    Value perm = createTosaI32Const(rewriter, op.getLoc(), permutation);
    Value transposed = rewriter
                           .create<tosa::TransposeOp>(
                               op.getLoc(), transposedType, reshaped, perm)
                           .getResult();

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, transposed,
        rewriter.getDenseI64ArrayAttr(resultType.getShape()));
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

    patterns.insert<QuantizeOp, DequantizeOp, DepthToSpaceOp>(context);
    // We insert a clamp to enforce non-standard TOSA dataypes. E.g. i6 signed
    // integer range described with an i8 value. However, in the case we use i8
    // and clamp to values of i8 (i.e. si8) then the clamp can be optimized away
    // and the following canonicalization will check/do that.
    tosa::ClampOp::getCanonicalizationPatterns(patterns, context);

    FrozenRewritePatternSet frozenSetOfPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(module, frozenSetOfPatterns))) {
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
