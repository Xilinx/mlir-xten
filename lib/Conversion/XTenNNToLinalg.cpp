// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

namespace xilinx::xten {
#define GEN_PASS_DECL_CONVERTXTENNNTOLINALG
#define GEN_PASS_DEF_CONVERTXTENNNTOLINALG
#include "xten/Conversion/Passes.h.inc"
} // namespace xilinx::xten

using namespace mlir;
using namespace amd::xten_nn;

namespace {

int64_t getRank(Value v) {
  return cast<ShapedType>(v.getType()).getRank();
}

int64_t getMaxRank(ValueRange operands) {
  int64_t maxRank = 0;
  for (Value operand : operands) {
    maxRank = std::max(maxRank, getRank(operand));
  }
  return maxRank;
}

bool isScalar(Value v) {
  return getRank(v) == 0;
}

Value getEmptyTensor(OpBuilder &b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                   dynSizes,
                                   cast<RankedTensorType>(type).getEncoding());
}

Value createFloatConstant(OpBuilder &b, Location loc, Type type, double value) {
  return b.create<arith::ConstantOp>(loc, b.getFloatAttr(type, value));
}

Value createIndexConstant(OpBuilder &b, Location loc, int64_t value) {
  return b.create<arith::ConstantIndexOp>(loc, value);
}

Value castIndexToFloat(OpBuilder &b, Location loc, Value value,
                       Type floatType) {
  const Value intValue =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), value);
  return b.create<arith::SIToFPOp>(loc, floatType, intValue);
}

Value castFloatToIndex(OpBuilder &b, Location loc, Value value) {
  const Value intValue = b.create<arith::FPToSIOp>(loc, b.getI64Type(), value);
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), intValue);
}

Value clampIndex(OpBuilder &b, Location loc, Value value, int64_t upper) {
  Value zero = createIndexConstant(b, loc, 0);
  Value upperValue = createIndexConstant(b, loc, upper);
  Value isBelowZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, value, zero);
  Value lowerClamped =
      b.create<arith::SelectOp>(loc, isBelowZero, zero, value);
  Value isAboveUpper = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, lowerClamped, upperValue);
  return b.create<arith::SelectOp>(loc, isAboveUpper, upperValue, lowerClamped);
}

Value convertValue(OpBuilder &b, Location loc, Value value, Type dstType) {
  const Type srcType = value.getType();
  if (srcType == dstType)
    return value;
  auto srcFloat = dyn_cast<FloatType>(srcType);
  auto dstFloat = dyn_cast<FloatType>(dstType);
  if (!srcFloat || !dstFloat)
    return nullptr;
  if (srcFloat.getWidth() < dstFloat.getWidth())
    return b.create<arith::ExtFOp>(loc, dstType, value);
  return b.create<arith::TruncFOp>(loc, dstType, value);
}

SmallVector<ReassociationIndices>
getKeepDimsReassociation(int64_t inputRank, ArrayRef<int64_t> keptAxes) {
  SmallVector<ReassociationIndices> reassociation;
  int64_t start = 0;
  for (int64_t axis : keptAxes) {
    ReassociationIndices group;
    for (int64_t i = start; i <= axis; ++i)
      group.push_back(i);
    reassociation.push_back(group);
    start = axis + 1;
  }
  if (start < inputRank && !reassociation.empty()) {
    for (int64_t i = start; i < inputRank; ++i)
      reassociation.back().push_back(i);
  }
  return reassociation;
}

// Elu(x) = x > 0 ? x : alpha * (exp(x) - 1)
Value mapEluOpToArithAndMathOps(EluOp op, ArrayRef<Type> /*resultTypes*/,
                                Value operand, OpBuilder *b) {
  Location loc = op->getLoc();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (!isa<FloatType>(elementType)) {
    return nullptr;
  }

  // Build: exp(x) - 1
  Value exp = b->create<::mlir::math::ExpOp>(loc, operand);
  Value one =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 1));
  Value sub = b->create<::mlir::arith::SubFOp>(loc, exp, one);
  Value alphaAsValue = b->create<mlir::arith::ConstantFloatOp>(
      loc, EluOpAdaptor(op).getAlpha(), cast<FloatType>(elementType));
  Value mul = b->create<::mlir::arith::MulFOp>(loc, alphaAsValue, sub);

  // Build: x > 0 ? x : alpha * (exp(x) - 1)
  Value zero =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 0));
  Value cmpOp =
      b->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, operand, zero);
  return b->create<arith::SelectOp>(loc, cmpOp, operand, mul);
}

template <typename SrcOpT,
          Value codegenFunc(SrcOpT, ArrayRef<Type>, Value, OpBuilder *)>
class ElementWiseOpToLinalg : public OpConversionPattern<SrcOpT> {
public:
  using OpConversionPattern<SrcOpT>::OpConversionPattern;
  using OpAdaptor = typename SrcOpT::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ValueRange inputs = adaptor.getOperands();
    auto resultTy = cast<ShapedType>(op.getOutput().getType());
    Value output = getEmptyTensor(rewriter, loc, resultTy, {});

    int64_t maxRank = getMaxRank(adaptor.getOperands());

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(maxRank, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(maxRank);
    SmallVector<AffineMap> maps;
    for (Value v : inputs)
      maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);

    // Build `linalg.generic` op.
    bool failed = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? resultTy : TypeRange{}, inputs, output, maps,
        mlir::tosa::getNParallelLoopsAttrs(maxRank),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(output);
          Value innerResult =
              (*codegenFunc)(op, innerResultTy, args.front(), &rewriter);
          if (!innerResult) {
            failed = true;
          } else {
            nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
          }
        },
        linalg::getPrunedAttributeList(op));

    if (failed)
      return failure();

    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

using EluToLinalg = ElementWiseOpToLinalg<EluOp, mapEluOpToArithAndMathOps>;

// Lower ReduceMean as a sum reduction followed by an elementwise division.
// Keeping unit dimensions is modeled as a final reshape/fill step so the
// linalg reduction maps operate on the natural keepdims-free shape.
// Static shapes are required so the reduction divisor is a compile-time product
// of the reduced extents.
class ReduceMeanToLinalg : public OpConversionPattern<ReduceMeanOp> {
public:
  using OpConversionPattern<ReduceMeanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Location loc = op->getLoc();
    const Value input = adaptor.getInput();
    const auto inputTy = cast<RankedTensorType>(input.getType());
    const auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    const Type elementTy = inputTy.getElementType();
    const auto floatTy = dyn_cast<FloatType>(elementTy);
    if (!floatTy)
      return rewriter.notifyMatchFailure(
          op, "reduce_mean lowering only supports floating point tensors");
    if (!inputTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "reduce_mean lowering requires static input and result shapes");

    const int64_t rank = inputTy.getRank();
    llvm::SmallBitVector reduced(rank);
    for (int64_t axis : adaptor.getAxes()) {
      const int64_t normalized = axis < 0 ? axis + rank : axis;
      if (normalized < 0 || normalized >= rank)
        return rewriter.notifyMatchFailure(op, "axis is out of range");
      if (reduced.test(normalized))
        return rewriter.notifyMatchFailure(op, "duplicate axes are undefined");
      reduced.set(normalized);
    }

    if (adaptor.getAxes().empty()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    SmallVector<int64_t> reducedShape;
    SmallVector<int64_t> keptAxes;
    for (int64_t i = 0; i < rank; ++i) {
      if (reduced.test(i))
        continue;
      reducedShape.push_back(inputTy.getDimSize(i));
      keptAxes.push_back(i);
    }

    const auto reducedTy = RankedTensorType::get(reducedShape, elementTy);
    const Value reducedEmpty = getEmptyTensor(rewriter, loc, reducedTy, {});
    const Value zero = createFloatConstant(rewriter, loc, elementTy, 0.0);
    const Value filled =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                        ValueRange{reducedEmpty})
            .result();

    // Reduce to the keepdims-free shape first. Unit dimensions are restored
    // after the divide, which keeps the reduction maps simple.
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<AffineExpr> inputExprs;
    SmallVector<AffineExpr> outputExprs;
    inputExprs.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      AffineExpr expr = getAffineDimExpr(i, rewriter.getContext());
      inputExprs.push_back(expr);
      if (reduced.test(i)) {
        iteratorTypes[i] = utils::IteratorType::reduction;
      } else {
        outputExprs.push_back(expr);
      }
    }

    SmallVector<AffineMap> reductionMaps = AffineMap::inferFromExprList(
        {inputExprs, outputExprs}, rewriter.getContext());
    auto sum = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{reducedTy}, input, filled, reductionMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          const Value add =
              nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });

    // Mean is sum divided by the product of all reduced dimension sizes.
    Value divisorIndex = createIndexConstant(rewriter, loc, 1);
    for (int64_t i = 0; i < rank; ++i) {
      if (!reduced.test(i))
        continue;
      const Value dim =
          inputTy.isDynamicDim(i)
              ? rewriter.create<tensor::DimOp>(loc, input, i).getResult()
              : createIndexConstant(rewriter, loc, inputTy.getDimSize(i));
      divisorIndex = rewriter.create<arith::MulIOp>(loc, divisorIndex, dim);
    }
    const Value divisor =
        castIndexToFloat(rewriter, loc, divisorIndex, elementTy);

    Value divided;
    if (reducedTy.getRank() == 0) {
      const Value scalar = rewriter.create<tensor::ExtractOp>(
          loc, sum.getResult(0), ValueRange{});
      const Value div = rewriter.create<arith::DivFOp>(loc, scalar, divisor);
      divided = rewriter.create<tensor::FromElementsOp>(
          loc, reducedTy, ValueRange{div});
    } else {
      const Value divEmpty = getEmptyTensor(rewriter, loc, reducedTy, {});
      const AffineMap idMap =
          rewriter.getMultiDimIdentityMap(reducedTy.getRank());
      auto divOp = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{reducedTy}, sum.getResult(0), divEmpty,
          SmallVector<AffineMap>{idMap, idMap},
          mlir::tosa::getNParallelLoopsAttrs(reducedTy.getRank()),
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            const Value div =
                nestedBuilder.create<arith::DivFOp>(nestedLoc, args[0], divisor);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, div);
          });
      divided = divOp.getResult(0);
    }

    if (!adaptor.getKeepdims()) {
      rewriter.replaceOp(op, divided);
      return success();
    }

    if (keptAxes.empty()) {
      const Value scalar =
          rewriter.create<tensor::ExtractOp>(loc, divided, ValueRange{});
      const Value output = getEmptyTensor(rewriter, loc, resultTy, {});
      const Value filledOutput =
          rewriter.create<linalg::FillOp>(loc, ValueRange{scalar},
                                          ValueRange{output})
              .result();
      rewriter.replaceOp(op, filledOutput);
      return success();
    }

    const SmallVector<ReassociationIndices> reassociation =
        getKeepDimsReassociation(rank, keptAxes);
    const Value expanded =
        rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, divided,
                                               reassociation)
            .getResult();
    rewriter.replaceOp(op, expanded);
    return success();
  }
};

// Lower xten_nn.resize as one output-indexed linalg.generic. For every output
// element the body computes the corresponding real-valued input coordinate,
// samples the input with either nearest-neighbor or N-linear interpolation, and
// yields that sampled value.
//
// Supported cases:
//   * static rank-4 and rank-5 tensors, matching the op verifier;
//   * nearest mode for integer and floating element types;
//   * linear mode for floating element types;
//   * all xten_nn coordinate transformation modes and nearest modes.
//
class ResizeToLinalg : public OpConversionPattern<ResizeOp> {
public:
  using OpConversionPattern<ResizeOp>::OpConversionPattern;

  struct ResizeParams {
    Value input;
    RankedTensorType inputTy;
    RankedTensorType resultTy;
    ArrayRef<float> scales;
    uint64_t coordinateTransformationMode;
    uint64_t nearestMode;
  };

  // Map an output coordinate to the corresponding real-valued input
  // coordinate using the coordinate transformation formulas supported by
  // xten_nn.resize.
  static Value getCoordinate(const ResizeParams &params, OpBuilder &b,
                             Location loc, int64_t dim, Value outputIndex,
                             Type calcType) {
    const Value x = castIndexToFloat(b, loc, outputIndex, calcType);
    const Value scale =
        createFloatConstant(b, loc, calcType, params.scales[dim]);

    // coordinate_transformation_mode encoding:
    //   0: half_pixel          -> (x + 0.5) / scale - 0.5
    //   1: pytorch_half_pixel  -> same as half_pixel, except output size 1
    //                             maps to input coordinate 0
    //   2: asymmetric          -> x / scale
    //   3: align_corners       -> x * (input - 1) / (output - 1), except
    //                             output size 1 maps to input coordinate 0
    switch (params.coordinateTransformationMode) {
    case 0: {
      const Value half = createFloatConstant(b, loc, calcType, 0.5);
      const Value shifted = b.create<arith::AddFOp>(loc, x, half);
      const Value scaled = b.create<arith::DivFOp>(loc, shifted, scale);
      return b.create<arith::SubFOp>(loc, scaled, half);
    }
    case 1: {
      if (params.resultTy.getDimSize(dim) == 1)
        return createFloatConstant(b, loc, calcType, 0.0);
      const Value half = createFloatConstant(b, loc, calcType, 0.5);
      const Value shifted = b.create<arith::AddFOp>(loc, x, half);
      const Value scaled = b.create<arith::DivFOp>(loc, shifted, scale);
      return b.create<arith::SubFOp>(loc, scaled, half);
    }
    case 2:
      return b.create<arith::DivFOp>(loc, x, scale);
    case 3: {
      if (params.resultTy.getDimSize(dim) == 1)
        return createFloatConstant(b, loc, calcType, 0.0);
      const Value inMinusOne =
          createFloatConstant(b, loc, calcType,
                              params.inputTy.getDimSize(dim) - 1);
      const Value outMinusOne =
          createFloatConstant(b, loc, calcType,
                              params.resultTy.getDimSize(dim) - 1);
      const Value numerator = b.create<arith::MulFOp>(loc, x, inMinusOne);
      return b.create<arith::DivFOp>(loc, numerator, outMinusOne);
    }
    default:
      llvm_unreachable("unexpected resize coordinate transformation mode");
    }
  }

  // Convert a real-valued source coordinate to the nearest sampled input index,
  // then clamp to the input extent to match ONNX edge-padding behavior.
  static Value getNearestIndex(const ResizeParams &params, OpBuilder &b,
                               Location loc, int64_t dim, Value coord,
                               Type calcType) {
    const Value floorCoord = b.create<math::FloorOp>(loc, coord);
    const Value floorIndex = castFloatToIndex(b, loc, floorCoord);
    Value selectedIndex = floorIndex;

    if (params.nearestMode == 1 || params.nearestMode == 2) {
      const Value fraction = b.create<arith::SubFOp>(loc, coord, floorCoord);
      const Value half = createFloatConstant(b, loc, calcType, 0.5);
      const arith::CmpFPredicate pred = params.nearestMode == 1
                                            ? arith::CmpFPredicate::OGE
                                            : arith::CmpFPredicate::OGT;
      const Value takeUpper =
          b.create<arith::CmpFOp>(loc, pred, fraction, half);
      const Value one = createIndexConstant(b, loc, 1);
      const Value upperIndex = b.create<arith::AddIOp>(loc, floorIndex, one);
      selectedIndex =
          b.create<arith::SelectOp>(loc, takeUpper, upperIndex, floorIndex);
    }

    return clampIndex(b, loc, selectedIndex,
                      params.inputTy.getDimSize(dim) - 1);
  }

  static Value buildNearest(const ResizeParams &params, OpBuilder &b,
                            Location loc, Type calcType) {
    SmallVector<Value> indices;
    for (int64_t dim = 0; dim < params.inputTy.getRank(); ++dim) {
      const Value outIndex = b.create<linalg::IndexOp>(loc, dim);
      const Value coord =
          getCoordinate(params, b, loc, dim, outIndex, calcType);
      indices.push_back(getNearestIndex(params, b, loc, dim, coord, calcType));
    }
    return b.create<tensor::ExtractOp>(loc, params.input, indices);
  }

  static Value buildLinear(const ResizeParams &params, OpBuilder &b,
                           Location loc, Type calcType,
                           Type resultElementTy) {
    const int64_t rank = params.inputTy.getRank();

    SmallVector<Value> lowIndices;
    SmallVector<Value> highIndices;
    SmallVector<Value> lowWeights;
    SmallVector<Value> highWeights;
    const Value oneFloat = createFloatConstant(b, loc, calcType, 1.0);
    const Value oneIndex = createIndexConstant(b, loc, 1);

    // Precompute low/high source samples and interpolation weights for each
    // output dimension, then combine all 2^rank corners below.
    for (int64_t dim = 0; dim < rank; ++dim) {
      const Value outIndex = b.create<linalg::IndexOp>(loc, dim);
      const Value coord =
          getCoordinate(params, b, loc, dim, outIndex, calcType);
      const Value floorCoord = b.create<math::FloorOp>(loc, coord);
      const Value lowIndex = castFloatToIndex(b, loc, floorCoord);
      const Value highIndex = b.create<arith::AddIOp>(loc, lowIndex, oneIndex);
      const Value highWeight =
          b.create<arith::SubFOp>(loc, coord, floorCoord);
      const Value lowWeight =
          b.create<arith::SubFOp>(loc, oneFloat, highWeight);
      lowIndices.push_back(
          clampIndex(b, loc, lowIndex, params.inputTy.getDimSize(dim) - 1));
      highIndices.push_back(
          clampIndex(b, loc, highIndex, params.inputTy.getDimSize(dim) - 1));
      lowWeights.push_back(lowWeight);
      highWeights.push_back(highWeight);
    }

    Value acc = createFloatConstant(b, loc, calcType, 0.0);
    for (int64_t mask = 0, e = 1LL << rank; mask < e; ++mask) {
      SmallVector<Value> extractIndices;
      Value weight = createFloatConstant(b, loc, calcType, 1.0);
      for (int64_t dim = 0; dim < rank; ++dim) {
        const bool useHigh = mask & (1LL << dim);
        extractIndices.push_back(useHigh ? highIndices[dim] : lowIndices[dim]);
        weight = b.create<arith::MulFOp>(
            loc, weight, useHigh ? highWeights[dim] : lowWeights[dim]);
      }
      Value sample =
          b.create<tensor::ExtractOp>(loc, params.input, extractIndices);
      sample = convertValue(b, loc, sample, calcType);
      Value weighted = b.create<arith::MulFOp>(loc, sample, weight);
      acc = b.create<arith::AddFOp>(loc, acc, weighted);
    }

    return convertValue(b, loc, acc, resultElementTy);
  }

  LogicalResult
  matchAndRewrite(ResizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Location loc = op->getLoc();
    const Value input = adaptor.getX();
    const auto inputTy = cast<RankedTensorType>(input.getType());
    const auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    const Type elementTy = inputTy.getElementType();
    const ResizeParams params{input, inputTy, resultTy, adaptor.getScales(),
                              adaptor.getCoordinateTransformationMode(),
                              adaptor.getNearestMode()};

    if (!inputTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "resize lowering requires static input and result shapes");

    const bool isNearest = adaptor.getMode() == 0;
    const bool isLinear = adaptor.getMode() == 1;
    if (!isNearest && !isLinear)
      return rewriter.notifyMatchFailure(op, "unsupported resize mode");
    if (isLinear && !isa<FloatType>(elementTy))
      return rewriter.notifyMatchFailure(
          op, "linear resize lowering only supports floating point tensors");

    Type calcType = elementTy;
    if (auto floatTy = dyn_cast<FloatType>(elementTy)) {
      if (floatTy.getWidth() < 32)
        calcType = rewriter.getF32Type();
    } else {
      calcType = rewriter.getF32Type();
    }

    const Value output = getEmptyTensor(rewriter, loc, resultTy, {});
    const AffineMap outputMap =
        rewriter.getMultiDimIdentityMap(resultTy.getRank());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultTy}, ValueRange{}, output,
        SmallVector<AffineMap>{outputMap},
        mlir::tosa::getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange) {
          const Value result =
              isNearest
                  ? buildNearest(params, nestedBuilder, nestedLoc, calcType)
                  : buildLinear(params, nestedBuilder, nestedLoc, calcType,
                                elementTy);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

// Lower grid_sample to an output-indexed linalg.generic. Each output element
// extracts the normalized x/y grid coordinate, converts it to input-space pixel
// coordinates, samples the input with the requested interpolation rule, and
// applies zeros or border padding while materializing the sample.
class GridSampleToLinalg : public OpConversionPattern<GridSampleOp> {
public:
  using OpConversionPattern<GridSampleOp>::OpConversionPattern;

  struct GridSampleParams {
    Value input;
    Value grid;
    RankedTensorType inputTy;
    RankedTensorType gridTy;
    RankedTensorType resultTy;
    uint64_t alignCorners;
    uint64_t mode;
    uint64_t paddingMode;
  };

  static Value gridCoordinateToInput(const GridSampleParams &params,
                                     OpBuilder &b, Location loc,
                                     Value gridValue, int64_t dim,
                                     Type calcType) {
    gridValue = convertValue(b, loc, gridValue, calcType);
    const double inputSize = params.inputTy.getDimSize(dim);
    const Value one = createFloatConstant(b, loc, calcType, 1.0);
    const Value half = createFloatConstant(b, loc, calcType, 0.5);
    const Value shifted = b.create<arith::AddFOp>(loc, gridValue, one);

    if (params.alignCorners) {
      const Value sizeMinusOne =
          createFloatConstant(b, loc, calcType, inputSize - 1.0);
      const Value scaled = b.create<arith::MulFOp>(loc, shifted, sizeMinusOne);
      return b.create<arith::MulFOp>(loc, scaled, half);
    }

    const Value size = createFloatConstant(b, loc, calcType, inputSize);
    const Value scaled = b.create<arith::MulFOp>(loc, shifted, size);
    const Value shiftedBack = b.create<arith::SubFOp>(loc, scaled, one);
    return b.create<arith::MulFOp>(loc, shiftedBack, half);
  }

  static Value isIndexInBounds(OpBuilder &b, Location loc, Value index,
                               int64_t size) {
    const Value zero = createIndexConstant(b, loc, 0);
    const Value upper = createIndexConstant(b, loc, size - 1);
    const Value aboveLower =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, zero);
    const Value belowUpper =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, index, upper);
    return b.create<arith::AndIOp>(loc, aboveLower, belowUpper);
  }

  static Value sampleInput(const GridSampleParams &params, OpBuilder &b,
                           Location loc, Value n, Value c, Value h, Value w,
                           Type resultElementTy) {
    Value sampleH = h;
    Value sampleW = w;
    Value inBounds;
    if (params.paddingMode == 0) {
      Value hInBounds =
          isIndexInBounds(b, loc, h, params.inputTy.getDimSize(2));
      Value wInBounds =
          isIndexInBounds(b, loc, w, params.inputTy.getDimSize(3));
      inBounds = b.create<arith::AndIOp>(loc, hInBounds, wInBounds);
      sampleH = clampIndex(b, loc, h, params.inputTy.getDimSize(2) - 1);
      sampleW = clampIndex(b, loc, w, params.inputTy.getDimSize(3) - 1);
    } else {
      sampleH = clampIndex(b, loc, h, params.inputTy.getDimSize(2) - 1);
      sampleW = clampIndex(b, loc, w, params.inputTy.getDimSize(3) - 1);
    }

    Value sample =
        b.create<tensor::ExtractOp>(loc, params.input,
                                    ValueRange{n, c, sampleH, sampleW});
    if (params.paddingMode == 0) {
      const Value zero = createFloatConstant(b, loc, resultElementTy, 0.0);
      sample = b.create<arith::SelectOp>(loc, inBounds, sample, zero);
    }
    return sample;
  }

  static Value getNearestIndex(OpBuilder &b, Location loc, Value coord,
                               Type calcType) {
    const Value floorCoord = b.create<math::FloorOp>(loc, coord);
    const Value floorIndex = castFloatToIndex(b, loc, floorCoord);
    const Value one = createIndexConstant(b, loc, 1);
    const Value upperIndex = b.create<arith::AddIOp>(loc, floorIndex, one);

    const Value fraction = b.create<arith::SubFOp>(loc, coord, floorCoord);
    const Value half = createFloatConstant(b, loc, calcType, 0.5);
    const Value takeUpper =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, fraction, half);
    const Value isTie =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, fraction, half);

    const Value floorInt =
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), floorIndex);
    const Value two = b.create<arith::ConstantIntOp>(loc, 2, 64);
    const Value remainder = b.create<arith::RemSIOp>(loc, floorInt, two);
    const Value zero = b.create<arith::ConstantIntOp>(loc, 0, 64);
    const Value floorIsOdd =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero);
    const Value tieTakesUpper =
        b.create<arith::AndIOp>(loc, isTie, floorIsOdd);
    const Value selectUpper =
        b.create<arith::OrIOp>(loc, takeUpper, tieTakesUpper);
    return b.create<arith::SelectOp>(loc, selectUpper, upperIndex, floorIndex);
  }

  static Value buildNearest(const GridSampleParams &params, OpBuilder &b,
                            Location loc, Type calcType,
                            Type resultElementTy) {
    const Value n = b.create<linalg::IndexOp>(loc, 0);
    const Value c = b.create<linalg::IndexOp>(loc, 1);
    const Value outH = b.create<linalg::IndexOp>(loc, 2);
    const Value outW = b.create<linalg::IndexOp>(loc, 3);
    const Value zero = createIndexConstant(b, loc, 0);
    const Value one = createIndexConstant(b, loc, 1);
    const Value gridX =
        b.create<tensor::ExtractOp>(loc, params.grid,
                                    ValueRange{n, outH, outW, zero});
    const Value gridY =
        b.create<tensor::ExtractOp>(loc, params.grid,
                                    ValueRange{n, outH, outW, one});
    const Value inputX =
        gridCoordinateToInput(params, b, loc, gridX, 3, calcType);
    const Value inputY =
        gridCoordinateToInput(params, b, loc, gridY, 2, calcType);
    const Value x = getNearestIndex(b, loc, inputX, calcType);
    const Value y = getNearestIndex(b, loc, inputY, calcType);
    return sampleInput(params, b, loc, n, c, y, x, resultElementTy);
  }

  static Value buildBilinear(const GridSampleParams &params, OpBuilder &b,
                             Location loc, Type calcType,
                             Type resultElementTy) {
    const Value n = b.create<linalg::IndexOp>(loc, 0);
    const Value c = b.create<linalg::IndexOp>(loc, 1);
    const Value outH = b.create<linalg::IndexOp>(loc, 2);
    const Value outW = b.create<linalg::IndexOp>(loc, 3);
    const Value zeroIndex = createIndexConstant(b, loc, 0);
    const Value oneIndex = createIndexConstant(b, loc, 1);
    const Value gridX =
        b.create<tensor::ExtractOp>(loc, params.grid,
                                    ValueRange{n, outH, outW, zeroIndex});
    const Value gridY =
        b.create<tensor::ExtractOp>(loc, params.grid,
                                    ValueRange{n, outH, outW, oneIndex});
    const Value inputX =
        gridCoordinateToInput(params, b, loc, gridX, 3, calcType);
    const Value inputY =
        gridCoordinateToInput(params, b, loc, gridY, 2, calcType);

    const Value x0Float = b.create<math::FloorOp>(loc, inputX);
    const Value y0Float = b.create<math::FloorOp>(loc, inputY);
    const Value x0 = castFloatToIndex(b, loc, x0Float);
    const Value y0 = castFloatToIndex(b, loc, y0Float);
    const Value x1 = b.create<arith::AddIOp>(loc, x0, oneIndex);
    const Value y1 = b.create<arith::AddIOp>(loc, y0, oneIndex);

    const Value xLerp = b.create<arith::SubFOp>(loc, inputX, x0Float);
    const Value yLerp = b.create<arith::SubFOp>(loc, inputY, y0Float);
    const Value oneFloat = createFloatConstant(b, loc, calcType, 1.0);
    const Value x0Weight = b.create<arith::SubFOp>(loc, oneFloat, xLerp);
    const Value y0Weight = b.create<arith::SubFOp>(loc, oneFloat, yLerp);

    auto weightedSample = [&](Value h, Value w, Value hWeight,
                              Value wWeight) -> Value {
      Value sample = sampleInput(params, b, loc, n, c, h, w, resultElementTy);
      sample = convertValue(b, loc, sample, calcType);
      const Value weight = b.create<arith::MulFOp>(loc, hWeight, wWeight);
      return b.create<arith::MulFOp>(loc, sample, weight);
    };

    Value acc = weightedSample(y0, x0, y0Weight, x0Weight);
    acc = b.create<arith::AddFOp>(
        loc, acc, weightedSample(y0, x1, y0Weight, xLerp));
    acc = b.create<arith::AddFOp>(
        loc, acc, weightedSample(y1, x0, yLerp, x0Weight));
    acc = b.create<arith::AddFOp>(
        loc, acc, weightedSample(y1, x1, yLerp, xLerp));
    return convertValue(b, loc, acc, resultElementTy);
  }

  LogicalResult
  matchAndRewrite(GridSampleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Location loc = op->getLoc();
    const auto inputTy = cast<RankedTensorType>(adaptor.getX().getType());
    const auto gridTy = cast<RankedTensorType>(adaptor.getGrid().getType());
    const auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    const Type resultElementTy = resultTy.getElementType();

    if (!inputTy.hasStaticShape() || !gridTy.hasStaticShape() ||
        !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "grid_sample lowering requires static shapes");
    if (inputTy.getRank() != 4 || gridTy.getRank() != 4 ||
        resultTy.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "grid_sample lowering supports rank-4 tensors");
    if (!isa<FloatType>(inputTy.getElementType()) ||
        !isa<FloatType>(gridTy.getElementType()) ||
        !isa<FloatType>(resultElementTy))
      return rewriter.notifyMatchFailure(
          op, "grid_sample lowering only supports floating point tensors");
    // align_corners encoding: 0=false, 1=true.
    if (adaptor.getAlignCorners() > 1)
      return rewriter.notifyMatchFailure(
          op, "grid_sample align_corners must be 0 or 1");

    // mode encoding: 0=bilinear, 1=nearest, 2=cubic.
    if (adaptor.getMode() > 2)
      return rewriter.notifyMatchFailure(op, "invalid grid_sample mode");
    if (adaptor.getMode() == 2)
      return rewriter.notifyMatchFailure(
          op, "grid_sample cubic mode is not supported");

    // padding_mode encoding: 0=zeros, 1=border, 2=reflection.
    if (adaptor.getPaddingMode() > 2)
      return rewriter.notifyMatchFailure(
          op, "invalid grid_sample padding mode");
    if (adaptor.getPaddingMode() == 2)
      return rewriter.notifyMatchFailure(
          op, "grid_sample reflection padding is not supported");

    Type calcType = resultElementTy;
    if (auto floatTy = dyn_cast<FloatType>(calcType)) {
      if (floatTy.getWidth() < 32)
        calcType = rewriter.getF32Type();
    }

    const GridSampleParams params{adaptor.getX(),
                                  adaptor.getGrid(),
                                  inputTy,
                                  gridTy,
                                  resultTy,
                                  adaptor.getAlignCorners(),
                                  adaptor.getMode(),
                                  adaptor.getPaddingMode()};
    const Value output = getEmptyTensor(rewriter, loc, resultTy, {});
    const AffineMap outputMap =
        rewriter.getMultiDimIdentityMap(resultTy.getRank());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultTy}, ValueRange{}, output,
        SmallVector<AffineMap>{outputMap},
        mlir::tosa::getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange) {
          Value result = params.mode == 0
                             ? buildBilinear(params, nestedBuilder, nestedLoc,
                                             calcType, resultElementTy)
                             : buildNearest(params, nestedBuilder, nestedLoc,
                                            calcType, resultElementTy);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConvertXtenNNtoLinalg
    : public xilinx::xten::impl::ConvertXTenNNToLinalgBase<
          ConvertXtenNNtoLinalg> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, scf::SCFDialect, complex::ComplexDialect,
                math::MathDialect, shape::ShapeDialect, tensor::TensorDialect,
                arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    auto funcOp = getOperation();

    ConversionTarget target(*context);
    target.addIllegalOp<EluOp, GridSampleOp, ReduceMeanOp, ResizeOp, SignOp>();
    target.addLegalDialect<linalg::LinalgDialect, scf::SCFDialect,
                           complex::ComplexDialect, math::MathDialect,
                           shape::ShapeDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    RewritePatternSet patterns(context);
    patterns.add<EluToLinalg, GridSampleToLinalg, ReduceMeanToLinalg,
                 ResizeToLinalg>(context);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace amd {
namespace xten_nn {

std::unique_ptr<mlir::Pass> createXTenNNToLinalgPass() {
  return std::make_unique<ConvertXtenNNtoLinalg>();
}

} // namespace xten_nn
} // namespace amd
