// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "ATenDialect.h"
#include "ATenToStd.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
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
using namespace edsc::intrinsics;
using callOperation = edsc::OperationBuilder<mlir::CallOp>;
using call = edsc::ValueBuilder<mlir::CallOp>;
using constInt = edsc::intrinsics::std_constant_int;
using constFloat = edsc::intrinsics::std_constant_float;

namespace {

/// Utility function for type casting: this is making the type checker happy,
/// while delaying the actual work involved to convert the type. Most of the
/// time both side of the cast (producer and consumer) will be lowered to a
/// dialect like LLVM and end up with the same LLVM representation, at which
/// point this becomes a no-op and is eliminated.
Value typeCast(PatternRewriter &builder, Value val, Type destTy) {
  if (val.getType() == destTy)
    return val;
  return builder.create<xilinx::aten::TypeCastOp>(val.getLoc(), destTy, val)
      .getResult();
}

/// Create a type cast to memref
Value MemRefTypeCast(PatternRewriter &builder, Value val) {
  if (val.getType().isa<MemRefType>())
    return val;
  auto tensorTy = val.getType().dyn_cast<TensorType>();
  if (!tensorTy)
    return val;
  auto memRefType = mlir::MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), {}, 0);
  return typeCast(builder, val, memRefType);
}

std::string getMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
    ret << "M";
    auto shape = mrt.getShape();
    const Type elem = mrt.getElementType();
    for (auto s : shape)
      ret << s << "x";
    ret << getMangledType(elem);
  }
  else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  }
  else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  }
  else if (const xilinx::aten::ATenListType alt = ty.dyn_cast<const xilinx::aten::ATenListType>()) {

  }
  else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

std::string getMangledFuncName(ModuleOp module, std::string prefix, FunctionType fnTy) {
  std::string sep = "_";

  auto resultTy = fnTy.getResults();
  auto operTy = fnTy.getInputs();

  std::string ret = prefix;
  for (const Type t : resultTy)
    ret = ret + sep + getMangledType(t);
  for (const Type t : operTy)
    ret = ret + sep + getMangledType(t);

  return ret;
}

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value> operands, ArrayRef<Type> retTys)
{
  Builder builder(module);

  SmallVector<Type, 8> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix+"_AtenAcapOp", fnTy);

  auto fn = module.lookupSymbol<FuncOp>(fnName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    module.push_back(fn);
  }

  return fn;
}

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value> operands, Type &retTy)
{
  std::vector<Type> retTys{retTy};
  return getATenFn(module, prefix, operands, retTys);
}

/// Lower an aten.add to an affine loop nest.
///
/// This class inherit from `ConversionPattern` and override `rewrite`,
/// similarly to the PatternRewriter introduced in the previous chapter.
/// It will be called by the DialectConversion framework (see `ATenLowering`
/// class below).
class AddOpConversion_affine : public ConversionPattern {
public:
  explicit AddOpConversion_affine(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AddOp::getOperationName(), 1, context) {}

  /// Lower the `op` by generating IR using the `rewriter` builder. The builder
  /// is setup with a new function, the `operands` array has been populated with
  /// the rewritten operands for `op` in the new function.
  /// The results created by the new IR with the builder are returned, and their
  /// number must match the number of result of `op`.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto add = cast<xilinx::aten::AddOp>(op);
    auto loc = add.getLoc();
    Type resultTy = add.getResult().getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                      tensorResultTy.getElementType(),
                                                      {}, 0);

    //Value result = rewriter.create<xilinx::aten::AcapAllocOp>(loc, memRefResultTy, ArrayRef<Value>{});
    Value result = rewriter.create<AllocOp>(loc, memRefResultTy);
    Value lhs = MemRefTypeCast(rewriter, operands[0]);
    Value rhs = MemRefTypeCast(rewriter, operands[1]);
    auto indexType = IndexType::get(op->getContext());

    using namespace edsc;

    ScopedContext scope(rewriter, loc);
    Value zero = intrinsics::std_constant_index(0);
    Value one = intrinsics::std_constant_index(1);
    MemRefBoundsCapture vRes(result), vLHS(lhs), vRHS(rhs);
    StdIndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
    Value M(vRes.ub(0));
    if (vRes.rank() == 1) {
      Value i;
      AffineLoopNestBuilder(&i, {zero}, {M},
                      {one})(
          [&] { iRes(i) = iLHS(i) + iRHS(i); });
    } else if (vRes.rank() == 2) {
      Value N(vRes.ub(1));
      Value ivs[2];
      Value &i = ivs[0], &j = ivs[1];
      AffineLoopNestBuilder(ivs, {zero, zero}, {M, N},
                      {one, one})(
          [&] { iRes(i, j) = iLHS(i, j) + iRHS(i, j); });
    } else if (vRes.rank() == 3) {
      Value N(vRes.ub(1));
      Value O(vRes.ub(2));
      Value ivs[3];
      Value &i = ivs[0], &j = ivs[1], &k = ivs[2];
      AffineLoopNestBuilder(ivs, {zero, zero, zero}, {M, N, O},
                      {one, one, one})(
          [&] { iRes(i, j, k) = iLHS(i, j, k) + iRHS(i, j, k); });
    } else {
      Value N(vRes.ub(1));
      Value O(vRes.ub(2));
      Value P(vRes.ub(3));
      Value ivs[4];
      Value &i = ivs[0], &j = ivs[1], &k = ivs[2], &l = ivs[3];

      AffineLoopNestBuilder(ivs, {zero, zero, zero, zero}, {M, N, O, P},
                      {one, one, one, one})(
          [&] { iRes(i, j, k, l) = iLHS(i, j, k, l) + iRHS(i, j, k, l); });
    }
    // Return the newly allocated buffer.
    rewriter.replaceOp(op, {result});
    return success();
  }
};

/// Lower Add
class AddOpConversion : public ConversionPattern {
public:
  explicit AddOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AddOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);
    Value yVal(operands[1]);

    auto co = cast<xilinx::aten::ConstantOp>(operands[2].getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::AddmmOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value aVal(operands[0]);
    Value bVal(operands[1]);
    Value cVal(operands[2]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::AsStridedOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type elemTy = op->getResult(0).getType().cast<TensorType>().getElementType();

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

    // construct the shape argument
    std::vector<constInt> shape;
    std::vector<int64_t> result_shape;
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[1].getDefiningOp());
    DenseElementsAttr a0 = co0.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a0.getIntValues()) {
      shape.push_back(constInt(i.getSExtValue(),32));
      result_shape.push_back(i.getSExtValue());
    }

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1,32));

    // construct the stride argument
    std::vector<constInt> stride;
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[2].getDefiningOp());
    DenseElementsAttr a1 = co1.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a1.getIntValues())
      stride.push_back(constInt(i.getSExtValue(),32));

    // pad out the stride with -1 to make it 4d
    while (stride.size() < 4)
      stride.push_back(constInt(-1,32));

    APInt offset(32,0);
    if (operands.size() > 3) {
      auto co2 = cast<xilinx::aten::ConstantOp>(operands[3].getDefiningOp());
      auto ia2 = co2.getAttrOfType<IntegerAttr>("value");
      offset = ia2.getValue();
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
      : ConversionPattern(xilinx::aten::BatchNormOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value aVal(operands[0]);
    Value bVal(operands[1]);
    Value cVal(operands[2]);
    Value dVal(operands[3]);
    Value eVal(operands[4]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[5].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[6].getDefiningOp());
    auto fa0 = co1.getAttrOfType<FloatAttr>("value");
    APFloat faVal0 = fa0.getValue();

    auto co2 = cast<xilinx::aten::ConstantOp>(operands[7].getDefiningOp());
    auto fa1 = co2.getAttrOfType<FloatAttr>("value");
    APFloat faVal1 = fa1.getValue();

    auto co3 = cast<xilinx::aten::ConstantOp>(operands[8].getDefiningOp());
    auto ia1 = co3.getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    auto f32Ty = FloatType::getF32(op->getContext());

    std::vector<Value> callops{aVal, bVal, cVal, dVal, eVal,
                             constInt(iaVal0.getZExtValue(), 1),
                             constFloat(faVal0, f32Ty),
                             constFloat(faVal1, f32Ty),
                             constInt(iaVal1.getZExtValue(), 1)};

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
      : ConversionPattern(xilinx::aten::ConvolutionOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);
    Value wVal(operands[1]);
    Value bVal(operands[2]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[3], pad);
    unpack(operands[4], kernel);
    unpack(operands[5], stride);

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
      : ConversionPattern(xilinx::aten::ConvolutionBackwardOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]); // grad_output
    Value arg1(operands[1]); // input
    Value arg2(operands[2]); // weight

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[3], pad);
    unpack(operands[4], kernel);
    unpack(operands[5], stride);

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
      : ConversionPattern(xilinx::aten::DivOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);
    Value yVal(operands[1]);

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
      : ConversionPattern(xilinx::aten::LogSoftmaxOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value aVal(operands[0]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[1].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[2].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
class LogSoftmaxBackwardOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::LogSoftmaxBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);
    Value arg3(operands[3]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[2].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    std::vector<Value> callops{arg0, arg1,
                                constInt(iaVal0.getSExtValue(), 32),
                                arg3};

    FuncOp logsoftmaxBackwardFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                              "log_softmax_backward_data", callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                         rewriter.getSymbolRefAttr(logsoftmaxBackwardFunc),
                         callops);

    rewriter.replaceOp(op, (*new_call).getResults());
    return success();
  }
};

/// Lower maxpool2d
class MaxPoolOpConversion : public ConversionPattern {
public:
  explicit MaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MaxPool2dOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[1], kernel);
    unpack(operands[2], stride);
    unpack(operands[3], pad);

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
      : ConversionPattern(xilinx::aten::MaxPool2dWithIndicesOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride, dilation;
    unpack(operands[1], kernel);
    unpack(operands[2], stride);
    unpack(operands[3], pad);
    unpack(operands[4], dilation);

    //ceil_mode
    auto co = cast<xilinx::aten::ConstantOp>(operands[5].getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::MaxPool2dWithIndicesBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride, dilation;
    unpack(operands[2], kernel);
    unpack(operands[3], stride);
    unpack(operands[4], pad);
    unpack(operands[5], dilation);

    //ceil_mode
    auto co = cast<xilinx::aten::ConstantOp>(operands[6].getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::MMOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);
    Value yVal(operands[1]);

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
      : ConversionPattern(xilinx::aten::MulOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);
    Value yVal(operands[1]);

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
      : ConversionPattern(xilinx::aten::NativeBatchNormOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value aVal(operands[0]);
    Value bVal(operands[1]);
    Value cVal(operands[2]);
    Value dVal(operands[3]);
    Value eVal(operands[4]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[5].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[6].getDefiningOp());
    auto fa0 = co1.getAttrOfType<FloatAttr>("value");
    APFloat faVal0 = fa0.getValue();

    auto co2 = cast<xilinx::aten::ConstantOp>(operands[7].getDefiningOp());
    auto fa1 = co2.getAttrOfType<FloatAttr>("value");
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
      : ConversionPattern(xilinx::aten::NllLoss2dBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);
    Value arg2(operands[2]);
    Value arg3(operands[3]);
    Value arg6(operands[6]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[4].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[5].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::NllLoss2dForwardOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);
    Value arg2(operands[2]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::NllLossBackwardOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0).getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);
    Value arg2(operands[2]);
    Value arg3(operands[3]);
    Value arg6(operands[6]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[4].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[5].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::NllLossForwardOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);
    Value arg2(operands[2]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3].getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4].getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
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
      : ConversionPattern(xilinx::aten::ReLUOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

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
      : ConversionPattern(xilinx::aten::ThresholdBackwardOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value arg0(operands[0]);
    Value arg1(operands[1]);

    auto co = dyn_cast<xilinx::aten::ConstantOp>(operands[2].getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
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

/// Lower transpose
class TransposeOpConversion : public ConversionPattern {
public:
  explicit TransposeOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::TransposeOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

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
      : ConversionPattern(xilinx::aten::ViewOp::getOperationName(), 1, context) {}

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
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal(operands[0]);

    // construct the shape argument
    std::vector<constInt> shape;
    auto co = cast<xilinx::aten::ConstantOp>(operands[1].getDefiningOp());
    DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a.getIntValues())
      shape.push_back(constInt(i.getSExtValue(),32));

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

class NoOpConversion_affine : public ConversionPattern {
public:
  explicit NoOpConversion_affine(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AcapNoOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto noop = cast<xilinx::aten::AcapNoOp>(op);
    auto loc = noop.getLoc();
    Type resultTy = noop.getResult().getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                      tensorResultTy.getElementType(),
                                                      {}, 0);

    Value result = rewriter.create<AllocOp>(loc, memRefResultTy);
    Value lhs = MemRefTypeCast(rewriter, operands[0]);
    auto indexType = IndexType::get(op->getContext());

    using namespace edsc;

    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::std_constant_index(0);
    ValueHandle one = intrinsics::std_constant_index(1);
    MemRefBoundsCapture vRes(result);
    StdIndexedValue iRes(result), iLHS(lhs);
    ValueHandle i(indexType), j(indexType), k(indexType), l(indexType), M(vRes.ub(0));
    if (vRes.rank() == 1) {
      AffineLoopNestBuilder({&i}, {zero}, {M},
                      {one})([&] { iRes(i) = iLHS(i); });
    } else if (vRes.rank() == 2) {
      ValueHandle N(vRes.ub(1));
      AffineLoopNestBuilder({&i, &j}, {zero, zero}, {M, N},
                      {one, one})([&] { iRes(i, j) = iLHS(i, j); });
    } else if (vRes.rank() == 3) {
      ValueHandle N(vRes.ub(1));
      ValueHandle O(vRes.ub(2));

      AffineLoopNestBuilder({&i, &j, &k}, {zero, zero, zero}, {M, N, O},
                      {one, one, one})([&] { iRes(i, j, k) = iLHS(i, j, k); });
    }
    else {
      ValueHandle N(vRes.ub(1));
      ValueHandle O(vRes.ub(2));
      ValueHandle P(vRes.ub(3));

      AffineLoopNestBuilder({&i, &j, &k, &l}, {zero, zero, zero, zero}, {M, N, O, P},
                      {one, one, one, one})([&] { iRes(i, j, k, l) = iLHS(i, j, k, l); });
    }
    // Return the newly allocated buffer, with a type.cast to preserve the
    // consumers.
    rewriter.replaceOp(op, {result});
    return matchSuccess();
  }
};

/// Lower conv2d
class AcapConv2dReLUConversion : public ConversionPattern {
public:
  explicit AcapConv2dReLUConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AcapConv2dReLUOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);
    edsc::ValueHandle wVal(operands[1]);
    edsc::ValueHandle bVal(operands[2]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[3], pad);
    unpack(operands[4], kernel);
    unpack(operands[5], stride);

    auto padCI = constInt(pad[0],32);
    auto kernelCI = constInt(kernel[0], 32);
    auto strideCI = constInt(stride[0], 32);

    std::vector<Value> callops{xVal, wVal, bVal, padCI, kernelCI, strideCI};

    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d_relu", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(convFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower conv2d
class AcapConv2dBatchNormReLUConversion : public ConversionPattern {
public:
  explicit AcapConv2dBatchNormReLUConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AcapConv2dBatchNormReLUOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0).getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    std::vector<Value> callops;
    // conv2d operands
    {
      edsc::ValueHandle xVal(operands[0]);
      edsc::ValueHandle wVal(operands[1]);
      edsc::ValueHandle bVal(operands[2]);

      auto unpack = [](auto &op, auto &v) -> void {
        auto co = cast<xilinx::aten::ConstantOp>(op.getDefiningOp());
        DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
        for (auto i : a.getIntValues())
          v.push_back(i.getSExtValue());
      };

      std::vector<uint64_t> pad, kernel, stride;
      unpack(operands[3], pad);
      unpack(operands[4], kernel);
      unpack(operands[5], stride);

      auto padCI = constInt(pad[0],32);
      auto kernelCI = constInt(kernel[0], 32);
      auto strideCI = constInt(stride[0], 32);
      std::vector<Value> cops{xVal, wVal, bVal, padCI, kernelCI, strideCI};
      for (auto o : cops) callops.push_back(o);
    }
    {
      edsc::ValueHandle bVal(operands[11+1]);
      edsc::ValueHandle cVal(operands[11+2]);
      edsc::ValueHandle dVal(operands[11+3]);
      edsc::ValueHandle eVal(operands[11+4]);

      auto co0 = cast<xilinx::aten::ConstantOp>(operands[11+5].getDefiningOp());
      auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
      APInt iaVal0 = ia0.getValue();

      auto co1 = cast<xilinx::aten::ConstantOp>(operands[11+6].getDefiningOp());
      auto fa0 = co1.getAttrOfType<FloatAttr>("value");
      APFloat faVal0 = fa0.getValue();

      auto co2 = cast<xilinx::aten::ConstantOp>(operands[11+7].getDefiningOp());
      auto fa1 = co2.getAttrOfType<FloatAttr>("value");
      APFloat faVal1 = fa1.getValue();

      auto f32Ty = FloatType::getF32(op->getContext());

      std::vector<Value> cops{bVal, cVal, dVal, eVal,
                              constInt(iaVal0.getZExtValue(), 1),
                              constFloat(faVal0, f32Ty),
                              constFloat(faVal1, f32Ty)};
      for (auto o : cops) callops.push_back(o);
    }
    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d_bn_relu", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(convFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Convert an ATen type, this gets called for block and region arguments, and
/// attributes.
MemRefType convertTensorType(TensorType tensor) {
  return mlir::MemRefType::get(tensor.getShape(), tensor.getElementType(), {}, 0);
}

struct ATenLoweringPass : public PassWrapper<ATenLoweringPass,
                                             OperationPass<ModuleOp>> {

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(getOperation().getContext());
    typeConverter.addConversion([&](Type type) {
      if (auto tensor = type.dyn_cast<TensorType>())
        return convertTensorType(tensor).cast<Type>();
      return type;
    });


    // c++ patterns
    atenPatterns.insert<AddOpConversion, ConvolutionOpConversion,
                        ReLUOpConversion, TransposeOpConversion,
                        BatchNormOpConversion, NativeBatchNormOpConversion,
                        MaxPoolOpConversion, MaxPool2dWithIndicesOpConversion,
                        AddmmOpConversion, ViewOpConversion,
                        MulOpConversion, MMOpConversion,
                        AsStridedOpConversion, LogSoftmaxOpConversion,
                        ThresholdBackwardOpConversion, MaxPool2dWithIndicesBackwardOpConversion,
                        ConvolutionBackwardOpConversion, NllLossForwardOpConversion,
                        NllLossBackwardOpConversion, NllLoss2dForwardOpConversion,
                        NllLoss2dBackwardOpConversion, LogSoftmaxOpConversion,
                        LogSoftmaxBackwardOpConversion, DivOpConversion>(context);

    atenPatterns.insert<NoOpConversion_affine, AcapConv2dReLUConversion,
                        AcapConv2dBatchNormReLUConversion>(context);

    mlir::populateFuncOpTypeConversionPattern(atenPatterns,
                                              context,
                                              typeConverter);

    // tablegen patterns
    populateATenToStdPatterns(context, atenPatterns);

    // Perform aten specific lowering.
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           StandardOpsDialect, scf::SCFDialect>();
    target.addLegalOp<xilinx::aten::AcapAllocOp>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
       return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyPartialConversion(getModule(), target, atenPatterns, &typeConverter))) {
      emitError(UnknownLoc::get(context), "error lowering ATen\n");
      signalPassFailure();
    }

    // remove dead constant ops
    for (auto function : getOperation().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        auto constOp = dyn_cast<xilinx::aten::ConstantOp>(op);
        if (!constOp)
          return;
        if (op->use_empty())
          op->erase();
      });
    }
  }

};

}// namespace


namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenLoweringPass() {
  return std::make_unique<ATenLoweringPass>();
}

} // namespace aten
} // namespace xilinx
