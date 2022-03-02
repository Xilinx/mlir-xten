//===- ATenLoweringPass.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#if 0
#include "xten/Transform/ATenLoweringPass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "xten/Util/Util.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

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

/// Lower Add
class AddOpConversion : public ConversionPattern {
public:
  explicit AddOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::AddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));
    Value yVal(MemRefTypeCast(rewriter, operands[1]));

    auto co = operands[2].getDefiningOp<ConstantOp>();
    auto ia = co->getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    std::vector<Value> callops{xVal, yVal, constInt(iaVal.getSExtValue(), 32)};
    std::vector<Type> retTys{memRefResultTy};
    FuncOp addFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "add", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(addFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower Addmm
class AddmmOpConversion : public ConversionPattern {
public:
  explicit AddmmOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::AddmmOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value aVal(MemRefTypeCast(rewriter, operands[0]));
    Value bVal(MemRefTypeCast(rewriter, operands[1]));
    Value cVal(MemRefTypeCast(rewriter, operands[2]));

    auto co0 = operands[3].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = operands[4].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    std::vector<Value> callops{aVal, bVal, cVal,
                             constInt(iaVal0.getSExtValue(), 32),
                             constInt(iaVal1.getSExtValue(), 32)};

    FuncOp addmmFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                 "addmm", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(addmmFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower AsStrided
class AsStridedOpConversion : public ConversionPattern {
public:
  explicit AsStridedOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::AsStridedOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type elemTy = op->getResult(0).getType().cast<TensorType>().getElementType();

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    // construct the shape argument
    std::vector<constInt> shape;
    std::vector<int64_t> result_shape;
    unpack_int_list(operands[1], result_shape);

    for (auto i : result_shape) {
      shape.push_back(constInt(i,32));
    }

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1,32));

    // construct the stride argument
    std::vector<constInt> stride;
    std::vector<int64_t> stride_i;
    unpack_int_list(operands[2], stride_i);
    for (auto i : stride_i)
      stride.push_back(constInt(i,32));

    // pad out the stride with -1 to make it 4d
    while (stride.size() < 4)
      stride.push_back(constInt(-1,32));

    APInt offset(32,0);
    if (operands.size() > 3) {
      offset = APInt(32,operands[3].getDefiningOp<ConstantIntOp>().getValue());
    }

    std::vector<Value> callops{xVal,
                             shape[0], shape[1], shape[2], shape[3],
                             stride[0], stride[1], stride[2], stride[3],
                             constInt(offset.getSExtValue(), 32)};


    Type memRefResultTy = mlir::MemRefType::get(result_shape,
                                                elemTy,
                                                {}, 0);
    FuncOp asstridedFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "as_strided", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(asstridedFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower batchnorm
class BatchNormOpConversion : public ConversionPattern {
public:
  explicit BatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::BatchNormOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);
    TensorType meanTensorResultTy =
      op->getResult(1).getType().cast<TensorType>();
    Type meanResultTy =
      mlir::MemRefType::get(meanTensorResultTy.getShape(),
                            meanTensorResultTy.getElementType(),
                            {}, 0);
    TensorType invstdTensorResultTy =
      op->getResult(2).getType().cast<TensorType>();
    Type invstdResultTy =
      mlir::MemRefType::get(invstdTensorResultTy.getShape(),
                            invstdTensorResultTy.getElementType(),
                            {}, 0);

    auto loc = op->getLoc();

    Value aVal(MemRefTypeCast(rewriter, operands[0]));
    Value bVal(MemRefTypeCast(rewriter, operands[1]));
    Value cVal(MemRefTypeCast(rewriter, operands[2]));
    Value dVal(MemRefTypeCast(rewriter, operands[3]));
    Value eVal(MemRefTypeCast(rewriter, operands[4]));

    std::vector<Value> callops{aVal, bVal, cVal, dVal, eVal,
                             operands[5], operands[6], operands[7], operands[8]};

    auto resultTypes =  {memRefResultTy, meanResultTy, invstdResultTy};
    FuncOp batchnormFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "batch_norm", callops,
                                     resultTypes);

    auto new_call = callOperation(resultTypes,
                         rewriter.getSymbolRefAttr(batchnormFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower conv2d
class ConvolutionOpConversion : public ConversionPattern {
public:
  explicit ConvolutionOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::ConvolutionOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));
    Value wVal(MemRefTypeCast(rewriter, operands[1]));
    Value bVal(MemRefTypeCast(rewriter, operands[2]));

    std::vector<int64_t> pad, kernel, stride;
    unpack_int_list(operands[3], pad);
    unpack_int_list(operands[4], kernel);
    unpack_int_list(operands[5], stride);

    auto padCI = constInt(pad[0],32);
    auto kernelCI = constInt(kernel[0], 32);
    auto strideCI = constInt(stride[0], 32);

    std::vector<Value> callops{xVal, wVal, bVal, padCI, kernelCI, strideCI};

    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(convFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower conv2d backward
class ConvolutionBackwardOpConversion : public ConversionPattern {
public:
  explicit ConvolutionBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::ConvolutionBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0).getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);

    TensorType result1Ty = op->getResult(1).getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    TensorType result2Ty = op->getResult(2).getType().cast<TensorType>();
    Type memRefResult2Ty = mlir::MemRefType::get(result2Ty.getShape(),
                                                 result2Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0])); // grad_output
    Value arg1(MemRefTypeCast(rewriter, operands[1])); // input
    Value arg2(MemRefTypeCast(rewriter, operands[2])); // weight

    std::vector<int64_t> pad, kernel, stride;
    unpack_int_list(operands[3], pad);
    unpack_int_list(operands[4], kernel);
    unpack_int_list(operands[5], stride);

    auto padCI = constInt(pad[0],32);
    auto kernelCI = constInt(kernel[0], 32);
    auto strideCI = constInt(stride[0], 32);

    std::vector<Value> callops{arg0, arg1, arg2, padCI, kernelCI, strideCI};
    std::vector<mlir::Type> retTys{memRefResult0Ty, memRefResult1Ty, memRefResult2Ty};

    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d_backward", callops, retTys);

    auto new_call = callOperation(retTys,
                                  rewriter.getSymbolRefAttr(convFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower Div
class DivOpConversion : public ConversionPattern {
public:
  explicit DivOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::DivOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));
    Value yVal(MemRefTypeCast(rewriter, operands[1]));

    std::vector<Value> callops{xVal, yVal};

    FuncOp divFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "div", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(divFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower LogSoftmax
class LogSoftmaxOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::LogSoftmaxOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value aVal(MemRefTypeCast(rewriter, operands[0]));

    auto co0 = operands[1].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = operands[2].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    std::vector<Value> callops{aVal,
                                constInt(iaVal0.getSExtValue(), 32),
                                constInt(iaVal1.getZExtValue(), 1)};

    FuncOp logsoftmaxFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "log_softmax", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(logsoftmaxFunc),
                         callops);
    
    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower LogSoftmaxBackwardData
// class LogSoftmaxBackwardOpConversion : public ConversionPattern {
// public:
//   explicit LogSoftmaxBackwardOpConversion(MLIRContext *context)
//       : ConversionPattern(NPCOMP::aten::LogSoftmaxBackwardOp::getOperationName(), 1, context) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value > operands,
//                   ConversionPatternRewriter &rewriter) const override
//   {
//     TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
//     Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
//                                                 resultTy.getElementType(),
//                                                 {}, 0);

//     auto loc = op->getLoc();
//  

//     Value arg0(MemRefTypeCast(rewriter, operands[0]));
//     Value arg1(MemRefTypeCast(rewriter, operands[1]));
//     Value arg3(MemRefTypeCast(rewriter, operands[3]));

//     auto co0 = operands[2].getDefiningOp<ConstantOp>();
//     auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
//     APInt iaVal0 = ia0.getValue();

//     std::vector<Value> callops{arg0, arg1,
//                                 constInt(iaVal0.getSExtValue(), 32),
//                                 arg3};

//     FuncOp logsoftmaxBackwardFunc = getATenFn(op->getParentOfType<ModuleOp>(),
//                                               "log_softmax_backward_data", callops, memRefResultTy);

//     auto new_call = callOperation(memRefResultTy,
//                          rewriter.getSymbolRefAttr(logsoftmaxBackwardFunc),
//                          callops);

//     rewriter.replaceOp(op, (*new_call).getResults());
//     return success();
//   }
// };

/// Lower maxpool2d
class MaxPoolOpConversion : public ConversionPattern {
public:
  explicit MaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::MaxPool2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    std::vector<int64_t> pad, kernel, stride;
    unpack_int_list(operands[1], kernel);
    unpack_int_list(operands[2], stride);
    unpack_int_list(operands[3], pad);

    std::vector<Value> callops{xVal,
                                constInt(kernel[0],32),
                                constInt(stride[0],32),
                                constInt(pad[0],32)};

    FuncOp maxpoolFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                   "max_pool2d", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(maxpoolFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower maxpool2d
class MaxPool2dWithIndicesOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::MaxPool2dWithIndicesOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    Type idxTy = op->getResult(1).getType();
    TensorType tensorIdxTy = idxTy.cast<TensorType>();
    Type memRefIdxTy = mlir::MemRefType::get(tensorIdxTy.getShape(),
                                             tensorIdxTy.getElementType(),
                                             {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    std::vector<int64_t> pad, kernel, stride, dilation;
    unpack_int_list(operands[1], kernel);
    unpack_int_list(operands[2], stride);
    unpack_int_list(operands[3], pad);
    unpack_int_list(operands[4], dilation);

    //ceil_mode
    auto co = operands[5].getDefiningOp<ConstantOp>();
    auto ia = co->getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    std::vector<Value> callops{xVal,
                                constInt(kernel[0],32),
                                constInt(stride[0],32),
                                constInt(pad[0],32),
                                constInt(dilation[0],32),
                                constInt(iaVal.getZExtValue(), 1)};

    std::vector<mlir::Type> retTys{memRefResultTy, memRefIdxTy};

    FuncOp maxpoolFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                   "max_pool2d_with_indices", callops, retTys);

    auto new_call = callOperation(retTys,
                         rewriter.getSymbolRefAttr(maxpoolFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower max_pool2d_with_indicies_backward
class MaxPool2dWithIndicesBackwardOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::MaxPool2dWithIndicesBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    std::vector<int64_t> pad, kernel, stride, dilation;
    unpack_int_list(operands[2], kernel);
    unpack_int_list(operands[3], stride);
    unpack_int_list(operands[4], pad);
    unpack_int_list(operands[5], dilation);

    //ceil_mode
    auto co = operands[6].getDefiningOp<ConstantOp>();
    auto ia = co->getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    std::vector<Value> callops{operands[0], operands[1],
                                constInt(kernel[0],32),
                                constInt(stride[0],32),
                                constInt(pad[0],32),
                                constInt(dilation[0],32),
                                constInt(iaVal.getZExtValue(), 1),
                                operands[7]};

    FuncOp maxpoolbackFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                       "max_pool2d_with_indices_backward",
                                       callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                                  rewriter.getSymbolRefAttr(maxpoolbackFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower MM
class MMOpConversion : public ConversionPattern {
public:
  explicit MMOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::MmOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));
    Value yVal(MemRefTypeCast(rewriter, operands[1]));

    std::vector<Value> callops{xVal, yVal};

    FuncOp mmFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                              "mm", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(mmFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower Mul
class MulOpConversion : public ConversionPattern {
public:
  explicit MulOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::MulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));
    Value yVal(MemRefTypeCast(rewriter, operands[1]));

    std::vector<Value> callops{xVal, yVal};

    FuncOp mulFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "mul", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(mulFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower batchnorm
class NativeBatchNormOpConversion : public ConversionPattern {
public:
  explicit NativeBatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::NativeBatchNormOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);
    TensorType meanTensorResultTy =
      op->getResult(1).getType().cast<TensorType>();
    Type meanResultTy =
      mlir::MemRefType::get(meanTensorResultTy.getShape(),
                            meanTensorResultTy.getElementType(),
                            {}, 0);
    TensorType invstdTensorResultTy =
      op->getResult(2).getType().cast<TensorType>();
    Type invstdResultTy =
      mlir::MemRefType::get(invstdTensorResultTy.getShape(),
                            invstdTensorResultTy.getElementType(),
                            {}, 0);

    auto loc = op->getLoc();

    Value aVal(MemRefTypeCast(rewriter, operands[0]));
    Value bVal(MemRefTypeCast(rewriter, operands[1]));
    Value cVal(MemRefTypeCast(rewriter, operands[2]));
    Value dVal(MemRefTypeCast(rewriter, operands[3]));
    Value eVal(MemRefTypeCast(rewriter, operands[4]));

    auto co0 = operands[5].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = operands[6].getDefiningOp<ConstantOp>();
    auto fa0 = co1->getAttrOfType<FloatAttr>("value");
    APFloat faVal0 = fa0.getValue();

    auto co2 = operands[7].getDefiningOp<ConstantOp>();
    auto fa1 = co2->getAttrOfType<FloatAttr>("value");
    APFloat faVal1 = fa1.getValue();

    auto f32Ty = FloatType::getF32(op->getContext());

    std::vector<Value> callops{aVal, bVal, cVal, dVal, eVal,
                                constInt(iaVal0.getZExtValue(), 1),
                                constFloat(faVal0, f32Ty),
                                constFloat(faVal1, f32Ty)};

    auto resultTypes =  {memRefResultTy, meanResultTy, invstdResultTy};
    FuncOp batchnormFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "native_batch_norm", callops, resultTypes);

    auto new_call = callOperation(resultTypes,
                         rewriter.getSymbolRefAttr(batchnormFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// lower NLL Loss backward
class NllLoss2dBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::NllLoss2dBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0]));
    Value arg1(MemRefTypeCast(rewriter, operands[1]));
    Value arg2(MemRefTypeCast(rewriter, operands[2]));
    Value arg3(MemRefTypeCast(rewriter, operands[3]));
    Value arg6(MemRefTypeCast(rewriter, operands[6]));

    // reduction
    auto co0 = operands[4].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = operands[5].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt arg5 = ia1.getValue();

    std::vector<Value> callops{arg0, arg1, arg2, arg3,
                                constInt(arg4.getZExtValue(), 32),
                                constInt(arg5.getZExtValue(), 32),
                                arg6};

    FuncOp nllLoss2dFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                        "nll_loss2d_backward",
                                         callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                                  rewriter.getSymbolRefAttr(nllLoss2dFwdFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// lower NLL Loss forward
class NllLoss2dForwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dForwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::NllLoss2dForwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0).getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);
    TensorType result1Ty = op->getResult(0).getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0]));
    Value arg1(MemRefTypeCast(rewriter, operands[1]));
    Value arg2(MemRefTypeCast(rewriter, operands[2]));

    // reduction
    auto co0 = operands[3].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = operands[4].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia1.getValue();

    std::vector<Value> callops{arg0, arg1, arg2,
                                constInt(arg3.getZExtValue(), 32),
                                constInt(arg4.getZExtValue(), 32)};

    std::vector<Type> retTy{memRefResult0Ty,memRefResult1Ty};

    FuncOp nllLoss2dFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                        "nll_loss2d_forward",
                                         callops, retTy);

    auto new_call = callOperation(retTy,
                                  rewriter.getSymbolRefAttr(nllLoss2dFwdFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// lower NLL Loss backward
class NllLossBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLossBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::NllLossBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0]));
    Value arg1(MemRefTypeCast(rewriter, operands[1]));
    Value arg2(MemRefTypeCast(rewriter, operands[2]));
    Value arg3(MemRefTypeCast(rewriter, operands[3]));
    Value arg6(MemRefTypeCast(rewriter, operands[6]));

    // reduction
    auto co0 = operands[4].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = operands[5].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt arg5 = ia1.getValue();

    std::vector<Value> callops{arg0, arg1, arg2, arg3,
                                constInt(arg4.getZExtValue(), 32),
                                constInt(arg5.getZExtValue(), 32),
                                arg6};

    FuncOp nllLossFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "nll_loss_backward",
                                       callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                                  rewriter.getSymbolRefAttr(nllLossFwdFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// lower NLL Loss forward
class NllLossForwardOpConversion : public ConversionPattern {
public:
  explicit NllLossForwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::NllLossForwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0).getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);
    TensorType result1Ty = op->getResult(0).getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0]));
    Value arg1(MemRefTypeCast(rewriter, operands[1]));
    Value arg2(MemRefTypeCast(rewriter, operands[2]));

    // reduction
    auto co0 = operands[3].getDefiningOp<ConstantOp>();
    auto ia0 = co0->getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = operands[4].getDefiningOp<ConstantOp>();
    auto ia1 = co1->getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia1.getValue();

    std::vector<Value> callops{arg0, arg1, arg2,
                                constInt(arg3.getZExtValue(), 32),
                                constInt(arg4.getZExtValue(), 32)};

    std::vector<Type> retTy{memRefResult0Ty,memRefResult1Ty};

    FuncOp nllLossFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "nll_loss_forward",
                                       callops, retTy);

    auto new_call = callOperation(retTy,
                                  rewriter.getSymbolRefAttr(nllLossFwdFunc),
                                  callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower ReLU
class ReLUOpConversion : public ConversionPattern {
public:
  explicit ReLUOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::ReluOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    std::vector<Value> callops{xVal};

    FuncOp reluFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "relu", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(reluFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower ThresholdBackward
class ThresholdBackwardOpConversion : public ConversionPattern {
public:
  explicit ThresholdBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::ThresholdBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value arg0(MemRefTypeCast(rewriter, operands[0]));
    Value arg1(MemRefTypeCast(rewriter, operands[1]));

    auto co = operands[2].getDefiningOp<ConstantOp>();
    auto ia = co->getAttrOfType<IntegerAttr>("value");
    APInt arg2 = ia.getValue();

    std::vector<Value> callops{arg0, arg1,
                                constInt(arg2.getSExtValue(), 32)};

    FuncOp reluFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "threshold_backward",
                                callops,
                                memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(reluFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

// Lower transpose
class TransposeOpConversion : public ConversionPattern {
public:
  explicit TransposeOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::TOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    std::vector<Value> callops{xVal};

    FuncOp transposeFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "t", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(transposeFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower view
class ViewOpConversion : public ConversionPattern {
public:
  explicit ViewOpConversion(MLIRContext *context)
      : ConversionPattern(NPCOMP::aten::ViewOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();

    Value xVal(MemRefTypeCast(rewriter, operands[0]));

    // construct the shape argument
    std::vector<constInt> shape;
    auto co = cast<NPCOMP::Basicpy::BuildListOp>(operands[1].getDefiningOp());
    for (auto o : co.getOperation()->getOperands())
      shape.push_back(constInt(o.getDefiningOp<ConstantIntOp>().getValue(),32));

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1,32));

    std::vector<Value> callops{xVal, shape[0], shape[1], shape[2], shape[3]};

    FuncOp viewFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "view", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(viewFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getNumDims() == 1) {
      auto f = rewriter.create<AffineForOp>(op.getLoc(),
                                            op.getLowerBoundsOperands(),
                                            op.lowerBoundsMap(),
                                            op.getUpperBoundsOperands(),
                                            op.upperBoundsMap());
      f.region().getBlocks().clear();
      rewriter.inlineRegionBefore(op.region(), f.region(), f.region().end());
      rewriter.eraseOp(op);
      return success();
    }
    else if (op.getNumDims() == 2) {
      auto ub0 = op.upperBoundsMap().getResult(0).cast<AffineConstantExpr>();
      auto ub1 = op.upperBoundsMap().getResult(1).cast<AffineConstantExpr>();
      auto outer = rewriter.create<AffineForOp>(op.getLoc(), 0, ub0.getValue());
      auto outer_builder = OpBuilder::atBlockBegin(outer.getBody());
      auto inner = outer_builder.create<AffineForOp>(op.getLoc(), 0, ub1.getValue());
      auto ivs = op.getIVs();
      ivs[0].replaceAllUsesWith(outer.getInductionVar());
      ivs[1].replaceAllUsesWith(inner.getInductionVar());
      auto &body = op.getBody()->getOperations();
      inner.getBody()->getOperations().splice(inner.getBody()->begin(), body,
                                              body.begin(), --body.end());
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

class AllocOpLowering : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {

    auto memrefTy = op.getType();
    if (op.getType().getMemorySpace() != 0) {
      auto alloc = rewriter.create<memref::AllocOp>(op.getLoc(), MemRefType::get(memrefTy.getShape(),
                                            memrefTy.getElementType(), memrefTy.getAffineMaps(), 0));
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }
    return failure();
  }
};

class ReturnOpLowering : public OpRewritePattern<ReturnOp> {
public:
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {

    auto retTys = op.getOperandTypes();
    auto funcTy = op->getParentOfType<FuncOp>().getType();
    auto funcRetTys = funcTy.getResults();

    std::vector<Value> returnOperands;

    int idx = 0;
    for (auto a : llvm::zip(retTys, funcRetTys)) {
      if (std::get<0>(a).isa<TensorType>() && std::get<1>(a).isa<MemRefType>()) {
        auto oper = op.getOperand(idx);
        if (auto cast = dyn_cast_or_null<NPCOMP::aten::TypeCastOp>(oper.getDefiningOp())) {
          if (cast.getOperand().getType() == std::get<1>(a)) {
            returnOperands.push_back(cast.getOperand());
            rewriter.eraseOp(cast);
            continue;
          }
        }
        returnOperands.push_back(oper);
      }
      idx = idx + 1;
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, returnOperands);
    return success();
  }
};

class TypeCastOpLowering : public OpRewritePattern<NPCOMP::aten::TypeCastOp> {
public:
  using OpRewritePattern<NPCOMP::aten::TypeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NPCOMP::aten::TypeCastOp op,
                                PatternRewriter &rewriter) const override {

    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto resultTy = op.getResult().getType();
    auto oper = op.getOperand();

    // remove identity cast
    if (resultTy == oper.getType()) {
      rewriter.replaceOp(op, oper);
      return success();
    }
    // simplify casts of casts
    else if (oper.getDefiningOp()) {
      if (auto oper_cast = dyn_cast<NPCOMP::aten::TypeCastOp>(oper.getDefiningOp())) {
        if (resultTy == oper_cast.getOperand().getType())
          rewriter.replaceOp(op, oper_cast.getOperand());
        else
          rewriter.replaceOpWithNewOp<NPCOMP::aten::TypeCastOp>(op, op.getType(), oper_cast.getOperand());
        return success();
      }
    }
    return failure();
  }

};
/// Convert an ATen type, this gets called for block and region arguments, and
/// attributes.
MemRefType convertTensorType(TensorType tensor) {
  return mlir::MemRefType::get(tensor.getShape(), tensor.getElementType(), {}, 0);
}

struct ATenLoweringPass : public PassWrapper<ATenLoweringPass,
                                             OperationPass<ModuleOp>> {

   void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<mlir::AffineDialect>();
     registry.insert<mlir::LLVM::LLVMDialect>();
   }

   void runOnOperation() override {
    LLVMTypeConverter typeConverter(getOperation().getContext());
    typeConverter.addConversion([&](Type type) -> Type {
      if (auto tensor = type.dyn_cast<TensorType>())
        return convertTensorType(tensor).cast<Type>();
      // make all memory spaces zero
      if (auto memref = type.dyn_cast<MemRefType>())
        return mlir::MemRefType::get(memref.getShape(), memref.getElementType(), memref.getAffineMaps(), 0);
      return type;
    });

    RewritePatternSet atenPatterns(&getContext());
    auto module = getOperation();
    auto context = module.getContext();

    // c++ patterns
    atenPatterns.insert<AddOpConversion, ConvolutionOpConversion,
                        ReLUOpConversion, TransposeOpConversion,
                        BatchNormOpConversion, NativeBatchNormOpConversion,
                        MaxPoolOpConversion, MaxPool2dWithIndicesOpConversion,
                        AddmmOpConversion, ViewOpConversion,
                        MulOpConversion, MMOpConversion,
                        AsStridedOpConversion,
                        ThresholdBackwardOpConversion, MaxPool2dWithIndicesBackwardOpConversion,
                        ConvolutionBackwardOpConversion, NllLossForwardOpConversion,
                        NllLossBackwardOpConversion, NllLoss2dForwardOpConversion,
                        NllLoss2dBackwardOpConversion, LogSoftmaxOpConversion,
                        /*LogSoftmaxBackwardOpConversion,*/ DivOpConversion>(context);

    atenPatterns.insert<TypeCastOpLowering,
                        ReturnOpLowering,
                        AffineParallelLowering, AllocOpLowering>(context);

    mlir::populateFuncOpTypeConversionPattern(atenPatterns,
                                              typeConverter);

    // Perform aten specific lowering.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect,
                           StandardOpsDialect,
                           scf::SCFDialect>();
    target.addLegalOp<AffineApplyOp,
                      AffineForOp,
                      AffineLoadOp,
                      AffineStoreOp,
                      AffineYieldOp>();

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpace() == 0);
    });

    target.addDynamicallyLegalOp<ReturnOp>([&](ReturnOp op) {
      for (auto rty : op.getOperandTypes())
        if (!rty.isa<MemRefType>())
          return false;
      return true;
    });

    if (failed(applyPartialConversion(module, target, std::move(atenPatterns)))) {
      emitError(UnknownLoc::get(context), "error lowering ATen\n");
      signalPassFailure();
    }

    // remove dead constant ops
    for (auto function : getOperation().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        auto constOp = dyn_cast<ConstantOp>(op);
        if (!constOp)
          return;
        if (op->use_empty())
          op->erase();
      });
    }

    for (auto function : getOperation().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        auto tc = dyn_cast<NPCOMP::aten::TypeCastOp>(op);
        if (!tc)
          return;
        // remove noop casts
        if (tc.getType() == tc.getOperand().getType())
          tc.replaceAllUsesWith(tc.getOperand());
        // remove cast if the tensor was the load of a memref
        else if (auto o = tc.getOperand().getDefiningOp())
          if (auto tl = dyn_cast<memref::TensorLoadOp>(o))
            tc.replaceAllUsesWith(tl.getOperand());
        if (op->use_empty())
          op->erase();
      });
    }

  }

};

}// namespace


namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createATenLoweringPass() {
  return std::make_unique<ATenLoweringPass>();
}

} // namespace xten
} // namespace xilinx
#endif