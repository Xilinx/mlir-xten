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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "xten-to-linalg-pass"

using namespace mlir;
using namespace xilinx::xten;
using namespace mlir::torch;

static Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
                      Attribute padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type inputETy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();

  assert(inputShape.size() == pad.size());

  SmallVector<int64_t, 4> paddedShape;
  SmallVector<OpFoldResult, 8> lowIndices;
  SmallVector<OpFoldResult, 8> highIndices;
  for (size_t i = 0, s = inputShape.size(); i < s; i++) {
    auto lowPad = pad[i];
    auto highPad = pad[i];
    paddedShape.push_back(inputShape[i] + highPad + lowPad);
    lowIndices.push_back(rewriter.getIndexAttr(lowPad));
    highIndices.push_back(rewriter.getIndexAttr(highPad));
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, padAttr);

  return rewriter.create<tensor::PadOp>(
      loc, RankedTensorType::get(paddedShape, inputETy), input, lowIndices,
      highIndices, padValue);
}

/// Return a zero-initialized tensor of given size and dtype.
static Value zeroInit(ArrayRef<int64_t> sizes, mlir::Type elementType,
                      Location loc, ConversionPatternRewriter &rewriter) {
  Value initTensor =
      rewriter.create<tensor::EmptyOp>(loc, sizes, elementType);
  Value c0float = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));
  return rewriter.create<linalg::FillOp>(loc, c0float, initTensor).getResult(0);
}

/// Return an aten bias (vtensor or none) converted to a standard bias tensor.
static Value convertBias(Operation *op, Value atenBias, Location loc,
                         ConversionPatternRewriter &rewriter) {
  if (atenBias.getType().isa<Torch::NoneType>() ||
      atenBias.getType().isa<Torch::OptionalType>()) {
    auto resultTy =
        op->getResult(0).getType().cast<torch::Torch::BaseTensorType>();
    return zeroInit(resultTy.getSizes()[1], resultTy.getDtype(), loc, rewriter);
  }
  return ToBuiltinTensorTypeCast(rewriter, atenBias);
}

/// Produces an output-dimensioned tensor, initialized with an aten bias
/// (vtensor or none).
static Value getBiasedInit(Operation *op, Value atenBias, Location loc,
                           ConversionPatternRewriter &rewriter) {
  auto outputTy =
      op->getResult(0).getType().dyn_cast<torch::Torch::BaseTensorType>();
  assert(outputTy);
  auto elementType = outputTy.getDtype();
  Value initTensor = rewriter.create<tensor::EmptyOp>(
      loc, outputTy.getSizes(), elementType);

  if (atenBias.getType().isa<Torch::NoneType>()) {
    Value c0float = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    return rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
        .getResult(0);
  }
  auto bias = ToBuiltinTensorTypeCast(rewriter, atenBias);
  auto biasType = bias.getType().cast<RankedTensorType>();
  assert(biasType.getRank() == 1);
  assert(elementType == biasType.getElementType());

  auto resultRank = initTensor.getType().cast<RankedTensorType>().getRank();
  SmallVector<AffineMap> indexingMaps = {
      // bias is used to initialize the channels - dimension 1 of output
      AffineMap::get(/*dimCount=*/resultRank, /*symbolCount=*/0,
                     rewriter.getAffineDimExpr(1), rewriter.getContext()),
      rewriter.getMultiDimIdentityMap(resultRank)};
  SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
  return rewriter
      .create<linalg::GenericOp>(
          loc, initTensor.getType(), bias, initTensor, indexingMaps,
          iteratorTypes,
          [](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
          })
      .getResult(0);
}

template <class T>
static LogicalResult processConv2d(T &conv2dOp, Location &loc, Value &input,
                                   Type &elementType, Operation *op,
                                   ConversionPatternRewriter &rewriter) {
  if (!elementType.isa<mlir::FloatType>())
    return op->emitError("unimplemented: non-floating point type");

  SmallVector<int64_t> paddingInts;
  paddingInts.resize(2, 0);
  if (!matchPattern(conv2dOp.getPadding(),
                    Torch::m_TorchConstantIntList(paddingInts))) {
    return rewriter.notifyMatchFailure(op,
                                       "only support constant padding values");
  }

  /// paddedInput. input shape change based on padding
  Attribute zeroAttr = rewriter.getZeroAttr(elementType);
  input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

  int64_t groups;
  if (!matchPattern(conv2dOp.getGroups(), Torch::m_TorchConstantInt(&groups)))
    return rewriter.notifyMatchFailure(op, "only support constant int group");

  if (groups != 1)
    return op->emitError("Only support groups value '1'");

  return success();
}

namespace {

// Propagate the layer_name attribute from an XTen op to the
// corresponding LinAlg operation.
void propagateLayerName(Operation *srcOp, Operation *destOp) {
  auto attrVal = srcOp->getAttrOfType<StringAttr>("layer_name");
  if (attrVal) {
    destOp->setAttr(llvm::StringRef("layer_name"), attrVal);
  }
}

template <class T>
class XTenBinaryOpConversion : public ConversionPattern {
public:
  XTenBinaryOpConversion(StringRef rootName, PatternBenefit benefit,
                         MLIRContext *ctx)
      : ConversionPattern(rootName, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    Value A = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value B = ToBuiltinTensorTypeCast(rewriter, operands[1]);

    auto tensorTy = operands[0].getType().cast<Torch::BaseTensorType>();
    auto elementTy = tensorTy.getDtype();
    auto sizes = tensorTy.getSizes();
    auto rank = sizes.size();
    Value C = rewriter.create<tensor::EmptyOp>(loc, sizes, elementTy);

    SmallVector<Value, 2> inputTensors{A, B};
    SmallVector<Value, 1> outputTensors{C};

    auto identMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap, 4> indexMap(3, identMap);

    auto linalgOp =
        rewriter
            .create<linalg::GenericOp>(
                loc, C.getType(), inputTensors, outputTensors, indexMap,
                SmallVector<StringRef>(rank, getParallelIteratorTypeName()), "",
                static_cast<const T *>(this)->getDefaultLibraryFunc(),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange blockArgs) {
                  auto result = static_cast<const T *>(this)->emitBinaryOp(
                      op, elementTy, rewriter, blockArgs[0], blockArgs[1]);
                  nestedBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    propagateLayerName(op, linalgOp.getDefiningOp());

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, linalgOp, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
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

  Value emitBinaryOp(Operation *op, Type elementTy,
                     ConversionPatternRewriter &rewriter, Value a,
                     Value b) const {
    if (elementTy.isa<FloatType>())
      return rewriter.create<mlir::arith::AddFOp>(op->getLoc(), a, b);
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

  Value emitBinaryOp(Operation *op, Type elementTy,
                     ConversionPatternRewriter &rewriter, Value a,
                     Value b) const {
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<MMOp>(op);
    auto loc = mmult.getLoc();

    auto resultTy = op->getResult(0).getType();
    auto tTy = resultTy.cast<Torch::BaseTensorType>();
    auto oper0Ty = operands[0].getType().cast<Torch::BaseTensorType>();
    auto oper1Ty = operands[1].getType().cast<Torch::BaseTensorType>();
    std::vector<int64_t> sizes{oper0Ty.getSizes()[0], oper1Ty.getSizes()[1]};

    Value A = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value B = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value C = zeroInit(tTy.getSizes(), tTy.getDtype(), loc, rewriter);

    auto mulOp = rewriter
                     .create<linalg::MatmulOp>(loc, C.getType(),
                                               ValueRange{A, B}, ValueRange{C})
                     .getResult(0);
    propagateLayerName(op, mulOp.getDefiningOp());

    auto tensor_cast = ToTorchTensorTypeCast(rewriter, mulOp, resultTy);
    rewriter.replaceOp(op, tensor_cast);

    return success();
  }
};

class XTenConv2dOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2d = cast<Conv2dOp>(op);
    auto loc = conv2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value biasInitTensor = getBiasedInit(op, conv2d.getBias(), loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2d.getPadding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2d.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2d.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    int64_t groups;
    if (!matchPattern(conv2d.getGroups(), Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    Value conv2dVal =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, biasInitTensor.getType(), ValueRange{input, weight},
                biasInitTensor, stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dVal.getDefiningOp());

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, conv2dVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dReluOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dReluOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dRelu = cast<Conv2dReLUOp>(op);
    auto loc = conv2dRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    Value conv2dReluVal =
        rewriter
            .create<linalg::Conv2DReluOp>(loc, initTensor.getType(),
                                          ValueRange{input, weight, bias},
                                          initTensor, stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dReluVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLRelu = cast<Conv2dLReLUOp>(op);
    auto loc = conv2dLRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dLRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    Value conv2dLReluVal = rewriter
                               .create<linalg::Conv2DLreluOp>(
                                   loc, initTensor.getType(),
                                   ValueRange{input, weight, bias, alpha},
                                   initTensor, stridesAttr, dilationAttr)
                               .getResult(0);
    propagateLayerName(op, conv2dLReluVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dLReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2d = cast<Conv2dTensorAddOp>(op);
    auto loc = conv2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2d, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2d.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2d.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dVal =
        rewriter
            .create<linalg::Conv2DTensorAddOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias,
                           add_ifm}, // add_ifm should be in ValueRange
                initTensor, stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dVal.getDefiningOp());

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, conv2dVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddReLUOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddReLUOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddReLUOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dRelu = cast<Conv2dTensorAddReLUOp>(op);
    auto loc = conv2dRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dReluVal =
        rewriter
            .create<linalg::Conv2DTensorAddReluOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias,
                           add_ifm}, // add_ifm should be in ValueRange
                initTensor, stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dReluVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddLReLUOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddLReLUOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddLReLUOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLRelu = cast<Conv2dTensorAddLReLUOp>(op);
    auto loc = conv2dLRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dLRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[8]);

    // Change appropriate operation over here
    Value conv2dLReluVal =
        rewriter
            .create<linalg::Conv2DTensorAddLreluOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, add_ifm,
                           alpha}, // add_ifm should be in ValueRange
                initTensor, stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dLReluVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dLReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddGlobalAveragePoolOpConversion
    : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddGlobalAveragePoolOpConversion(
      MLIRContext *context)
      : ConversionPattern(
            Conv2dTensorAddGlobalAveragePoolOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2d = cast<Conv2dTensorAddGlobalAveragePoolOp>(op);
    auto loc = conv2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2d, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2d.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2d.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value addIfm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dAveragePoolVal =
        rewriter
            .create<linalg::Conv2DTensorAddGlobalaveragepoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, addIfm}, initTensor,
                stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dAveragePoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dAveragePoolVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddReLUGlobalAveragePoolOpConversion
    : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddReLUGlobalAveragePoolOpConversion(
      MLIRContext *context)
      : ConversionPattern(
            Conv2dTensorAddReLUGlobalAveragePoolOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dRelu = cast<Conv2dTensorAddReLUGlobalAveragePoolOp>(op);
    auto loc = conv2dRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value addIfm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dReluAveragePoolVal =
        rewriter
            .create<linalg::Conv2DTensorAddReluGlobalaveragepoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, addIfm}, initTensor,
                stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dReluAveragePoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dReluAveragePoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddLReLUGlobalAveragePoolOpConversion
    : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddLReLUGlobalAveragePoolOpConversion(
      MLIRContext *context)
      : ConversionPattern(
            Conv2dTensorAddLReLUGlobalAveragePoolOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLRelu = cast<Conv2dTensorAddLReLUGlobalAveragePoolOp>(op);
    auto loc = conv2dLRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result =
        processConv2d(conv2dLRelu, loc, input, elementType, op, rewriter);
    if (result.failed())
      return result;

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto addCst = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, addCst);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLRelu.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLRelu.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value addIfm = ToBuiltinTensorTypeCast(rewriter, operands[8]);

    // Change appropriate operation over here
    Value conv2dAddLReluAvgPoolVal =
        rewriter
            .create<linalg::Conv2DTensorAddLreluGlobalaveragepoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, addIfm, alpha}, initTensor,
                stridesAttr, dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dAddLReluAvgPoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dAddLReluAvgPoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUMaxPoolOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLReluMaxpool = cast<Conv2dLReLUMaxPoolOp>(op);
    auto loc = conv2dLReluMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dLReluMaxpool.getPadding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLReluMaxpool.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLReluMaxpool.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dLReluMaxpool.getMpKernelSize(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dLReluMaxpool.getMpStride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t, 2> mp_paddingInts;
    if (!matchPattern(conv2dLReluMaxpool.getMpPadding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dLReluMaxpool.getMpDilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dLReluMaxpool.getGroups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dLReluMaxpool.getMpCeilMode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> new_mp_paddingInts; // Hl, Hh, Wl, Wh
    for (uint64_t i = 0; i < mp_paddingInts.size(); i++) {
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
    }

    long new_mp_paddingInts_size = new_mp_paddingInts.size();
    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({new_mp_paddingInts_size}, rewriter.getI64Type()),
        new_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue =
        rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initValue, initTensor)
            .getResult(0);

    Value conv2dLReluMaxpoolVal =
        rewriter
            .create<linalg::Conv2DLreluMaxpoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, alpha}, filledInitTensor,
                stridesAttr, dilationAttr, mp_kernel_sizeAttr, mp_stridesAttr,
                mp_paddingAttr, mp_dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dLReluMaxpoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dLReluMaxpoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluPadMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluPadMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUPadMaxPoolOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLReluPadMaxpool = cast<Conv2dLReLUPadMaxPoolOp>(op);
    auto loc = conv2dLReluPadMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dLReluPadMaxpool.getPadding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLReluPadMaxpool.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLReluPadMaxpool.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t> pad_paddingInts; // Wl, Wh, Hl, Hh
    if (!matchPattern(conv2dLReluPadMaxpool.getPadPadding(),
                      Torch::m_TorchConstantIntList(pad_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad_padding");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dLReluPadMaxpool.getMpKernelSize(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dLReluPadMaxpool.getMpStride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t> mp_paddingInts; // H, W
    if (!matchPattern(conv2dLReluPadMaxpool.getMpPadding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dLReluPadMaxpool.getMpDilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dLReluPadMaxpool.getGroups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dLReluPadMaxpool.getMpCeilMode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> pad_mp_paddingInts; // Hl,Hh, Wl, Wh
    if (pad_paddingInts.size() != mp_paddingInts.size() * 2)
      return rewriter.notifyMatchFailure(
          op, "max_pool padding is not double of pad padding");

    for (uint64_t i = mp_paddingInts.size(); i > 0; --i) {
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 2] +
                                   mp_paddingInts[i - 1]);
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 1] +
                                   mp_paddingInts[i - 1]);
    }

    long pad_mp_padding_size = pad_mp_paddingInts.size();

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto pad_mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({pad_mp_padding_size}, rewriter.getI64Type()),
        pad_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue =
        rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initValue, initTensor)
            .getResult(0);

    Value conv2dLReluPadMaxpoolVal =
        rewriter
            .create<linalg::Conv2DLreluMaxpoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, alpha}, filledInitTensor,
                stridesAttr, dilationAttr, mp_kernel_sizeAttr, mp_stridesAttr,
                pad_mp_paddingAttr, mp_dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dLReluPadMaxpoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dLReluPadMaxpoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dReluMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dReluMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dReLUMaxPoolOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dReluMaxpool = cast<Conv2dReLUMaxPoolOp>(op);
    auto loc = conv2dReluMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dReluMaxpool.getPadding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dReluMaxpool.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dReluMaxpool.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dReluMaxpool.getMpKernelSize(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dReluMaxpool.getMpStride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t, 2> mp_paddingInts;
    if (!matchPattern(conv2dReluMaxpool.getMpPadding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dReluMaxpool.getMpDilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dReluMaxpool.getGroups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dReluMaxpool.getMpCeilMode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> new_mp_paddingInts; // Hl, Hh, Wl, Wh
    for (uint64_t i = 0; i < mp_paddingInts.size(); i++) {
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
    }

    long new_mp_paddingInts_size = new_mp_paddingInts.size();
    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({new_mp_paddingInts_size}, rewriter.getI64Type()),
        new_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue =
        rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initValue, initTensor)
            .getResult(0);

    Value conv2dReluMaxpoolVal =
        rewriter
            .create<linalg::Conv2DReluMaxpoolOp>(
                loc, initTensor.getType(), ValueRange{input, weight, bias},
                filledInitTensor, stridesAttr, dilationAttr, mp_kernel_sizeAttr,
                mp_stridesAttr, mp_paddingAttr, mp_dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dReluMaxpoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dReluMaxpoolVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dReluPadMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dReluPadMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dReLUPadMaxPoolOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dReluPadMaxpool = cast<Conv2dReLUPadMaxPoolOp>(op);
    auto loc = conv2dReluPadMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dReluPadMaxpool.getPadding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dReluPadMaxpool.getStride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dReluPadMaxpool.getDilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t> pad_paddingInts; // Wl, Wh, Hl, Hh
    if (!matchPattern(conv2dReluPadMaxpool.getPadPadding(),
                      Torch::m_TorchConstantIntList(pad_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad_padding");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dReluPadMaxpool.getMpKernelSize(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dReluPadMaxpool.getMpStride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t> mp_paddingInts; // H, W
    if (!matchPattern(conv2dReluPadMaxpool.getMpPadding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dReluPadMaxpool.getMpDilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dReluPadMaxpool.getGroups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dReluPadMaxpool.getMpCeilMode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> pad_mp_paddingInts; // Hl,Hh, Wl, Wh
    if (pad_paddingInts.size() != mp_paddingInts.size() * 2)
      return rewriter.notifyMatchFailure(
          op, "max_pool padding is not double of pad padding");

    for (uint64_t i = mp_paddingInts.size(); i > 0; --i) {
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 2] +
                                   mp_paddingInts[i - 1]);
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 1] +
                                   mp_paddingInts[i - 1]);
    }

    long pad_mp_padding_size = pad_mp_paddingInts.size();

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto pad_mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({pad_mp_padding_size}, rewriter.getI64Type()),
        pad_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue =
        rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initValue, initTensor)
            .getResult(0);

    Value conv2dReluPadMaxpoolVal =
        rewriter
            .create<linalg::Conv2DReluMaxpoolOp>(
                loc, initTensor.getType(), ValueRange{input, weight, bias},
                filledInitTensor, stridesAttr, dilationAttr, mp_kernel_sizeAttr,
                mp_stridesAttr, pad_mp_paddingAttr, mp_dilationAttr)
            .getResult(0);
    propagateLayerName(op, conv2dReluPadMaxpoolVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dReluPadMaxpoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenPartialConv2dReLUOpConversion : public ConversionPattern {
public:
  explicit XTenPartialConv2dReLUOpConversion(MLIRContext *context)
      : ConversionPattern(PartialConv2dReLUOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    /*
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

     rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B},
     ValueRange{C});

     auto tensor_cast = TensorTypeCast(rewriter, C, op->getResult(0).getType());

     if(mmult.getNumResults() == 1)
       rewriter.replaceOp(op, tensor_cast);
     else
       rewriter.replaceOp(op, {tensor_cast, operands[0]});
     */
    return failure();
  }
};

class XTenSoftmaxOpConversion : public ConversionPattern {
public:
  explicit XTenSoftmaxOpConversion(MLIRContext *context)
      : ConversionPattern(SoftmaxOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto softmax = cast<SoftmaxOp>(op);
    auto loc = softmax.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    auto torchDim = (operands[1].getDefiningOp<Torch::ConstantIntOp>()).value();
    auto dim = rewriter.getI64IntegerAttr(torchDim.getSExtValue());

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    Value softmaxVal =
        rewriter
            .create<linalg::SoftmaxOp>(loc, initTensor.getType(), input, dim)
            .getResult();
    propagateLayerName(op, softmaxVal.getDefiningOp());

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, softmaxVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenGlobalAveragePool2DOpConversion : public ConversionPattern {
public:
  explicit XTenGlobalAveragePool2DOpConversion(MLIRContext *context)
      : ConversionPattern(GlobalAveragePool2D::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto globalaveragepool2d = cast<GlobalAveragePool2D>(op);
    auto loc = globalaveragepool2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), elementType);

    Value globalavgVal = rewriter
                             .create<linalg::GlobalAveragePool2DOp>(
                                 loc, initTensor.getType(), input)
                             .getResult();
    propagateLayerName(op, globalavgVal.getDefiningOp());

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, globalavgVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenLinearOpConversion : public OpConversionPattern<LinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LinearOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Cast the input arguments to default tensor type and convert bias
    // to an empty tensor if not given.
    Value input = ToBuiltinTensorTypeCast(rewriter, op.getInput());
    Value weights = ToBuiltinTensorTypeCast(rewriter, op.getWeight());
    Value bias = convertBias(op, op.getBias(), loc, rewriter);

    // Create the linalg version of linear
    auto torchResultType =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultType = RankedTensorType::get(torchResultType.getSizes(),
                                            torchResultType.getDtype());
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    Value linearVal =
        rewriter
            .create<linalg::LinearOp>(
                loc, resultType, ValueRange{input, weights, bias}, initTensor)
            .getResult(0);
    propagateLayerName(op, linearVal.getDefiningOp());

    // We need to convert from the default tensor type to torch tensor before
    // replacing
    auto resultTorchTensorCast =
        ToTorchTensorTypeCast(rewriter, linearVal, op->getResult(0).getType());

    rewriter.replaceOp(op, resultTorchTensorCast);

    return success();
  }
};

class XTenToLinalgPass : public XTenToLinalgBase<XTenToLinalgPass> {

public:
  XTenToLinalgPass() = default;
  XTenToLinalgPass(const XTenToLinalgPass &pass){};

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    RewritePatternSet patterns(context);

    patterns.insert<
        XTenAddOpConversion, XTenMulOpConversion, XTenMMOpConversion,
        XTenConv2dOpConversion, XTenConv2dReluOpConversion,
        XTenConv2dLeakyReluOpConversion, XTenConv2dLeakyReluMaxPoolOpConversion,
        XTenConv2dLeakyReluPadMaxPoolOpConversion,
        XTenConv2dReluMaxPoolOpConversion, XTenConv2dReluPadMaxPoolOpConversion,
        XTenPartialConv2dReLUOpConversion, XTenConv2dTensorAddOpConversion,
        XTenConv2dTensorAddReLUOpConversion,
        XTenConv2dTensorAddLReLUOpConversion, XTenSoftmaxOpConversion,
        XTenGlobalAveragePool2DOpConversion,
        XTenConv2dTensorAddGlobalAveragePoolOpConversion,
        XTenConv2dTensorAddReLUGlobalAveragePoolOpConversion,
        XTenConv2dTensorAddLReLUGlobalAveragePoolOpConversion,
        XTenLinearOpConversion>(context);

    ConversionTarget target(*context);

    target.addIllegalDialect<XTenDialect>();
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           scf::SCFDialect, tensor::TensorDialect,
                           Torch::TorchDialect,
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
