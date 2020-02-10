// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "ATenDialect.h"
#include "ATenToStd.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
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
//using namespace edsc;
using callOperation = edsc::intrinsics::OperationBuilder<mlir::CallOp>;
using call = edsc::intrinsics::ValueBuilder<mlir::CallOp>;
using constInt = edsc::intrinsics::constant_int;
using constFloat = edsc::intrinsics::constant_float;

namespace {

/// Utility function for type casting: this is making the type checker happy,
/// while delaying the actual work involved to convert the type. Most of the
/// time both side of the cast (producer and consumer) will be lowered to a
/// dialect like LLVM and end up with the same LLVM representation, at which
/// point this becomes a no-op and is eliminated.
Value *typeCast(PatternRewriter &builder, Value *val, Type destTy) {
  if (val->getType() == destTy)
    return val;
  return builder.create<xilinx::aten::TypeCastOp>(val->getLoc(), destTy, val)
      .getResult();
}

/// Create a type cast to memref
Value *MemRefTypeCast(PatternRewriter &builder, Value *val) {
  if (val->getType().isa<MemRefType>())
    return val;
  auto tensorTy = val->getType().dyn_cast<TensorType>();
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

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value *> &operands, ArrayRef<Type> &retTys)
{
  Builder builder(module);

  SmallVector<Type, 8> tys;
  for (auto o : operands)
    tys.push_back(o->getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix+"_AtenAcapOp", fnTy);

  auto fn = module.lookupSymbol<FuncOp>(fnName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    module.push_back(fn);
  }

  return fn;
}

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value *> &operands, Type &retTy)
{
  ArrayRef<Type> retTys{retTy};
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
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto add = cast<xilinx::aten::AddOp>(op);
    auto loc = add.getLoc();
    // Create an `aten.alloc` operation to allocate the output buffer for this op.
    Value *result = MemRefTypeCast(
        rewriter, rewriter.create<xilinx::aten::AllocOp>(loc, add.getResult()->getType())
                      .getResult());
    Value *lhs = MemRefTypeCast(rewriter, operands[0]);
    Value *rhs = MemRefTypeCast(rewriter, operands[1]);

    using namespace edsc;

    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::constant_index(0);
    ValueHandle one = intrinsics::constant_index(1);
    MemRefView vRes(result), vLHS(lhs), vRHS(rhs);
    IndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
    IndexHandle i, j, k, M(vRes.ub(0));
    ValueHandle *pi = &i;
    if (vRes.rank() == 1) {
      LoopNestBuilder({pi}, {zero}, {M},
                      {one})([&] { iRes(i) = iLHS(i) + iRHS(i); });
    } else if (vRes.rank() == 2) {
      IndexHandle N(vRes.ub(1));
      LoopNestBuilder({&i, &j}, {zero, zero}, {M, N},
                      {one, one})([&] { iRes(i, j) = iLHS(i, j) + iRHS(i, j); });
    } else {
        assert(vRes.rank() == 3 && "only ranks <= 3 are supported right now");
        IndexHandle N(vRes.ub(1));
        IndexHandle O(vRes.ub(2));

      LoopNestBuilder({&i, &j, &k}, {zero, zero, zero}, {M, N, O},
                      {one, one, one})([&] { iRes(i, j, k) = iLHS(i, j, k) + iRHS(i, j, k); });
    }

    // Return the newly allocated buffer, with a type.cast to preserve the
    // consumers.
    rewriter.replaceOp(op, {typeCast(rewriter, result, add.getType())});
    return matchSuccess();
  }
};

/// Lower Add
class AddOpConversion : public ConversionPattern {
public:
  explicit AddOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AddOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);
    edsc::ValueHandle yVal(operands[1]);

    auto co = cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    ArrayRef<Value*> callops{xVal, yVal, constInt(iaVal.getSExtValue(), 32)};
    ArrayRef<Type> retTys{memRefResultTy};
    FuncOp addFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "add", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(addFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower Addmm
class AddmmOpConversion : public ConversionPattern {
public:
  explicit AddmmOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AddmmOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle aVal(operands[0]);
    edsc::ValueHandle bVal(operands[1]);
    edsc::ValueHandle cVal(operands[2]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    ArrayRef<Value*> callops{aVal, bVal, cVal,
                             constInt(iaVal0.getSExtValue(), 32),
                             constInt(iaVal1.getSExtValue(), 32)};

    FuncOp addmmFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                 "addmm", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(addmmFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower AsStrided
class AsStridedOpConversion : public ConversionPattern {
public:
  explicit AsStridedOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::AsStridedOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    // construct the shape argument
    std::vector<constInt> shape;
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[1]->getDefiningOp());
    DenseElementsAttr a0 = co0.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a0.getIntValues())
      shape.push_back(constInt(i.getSExtValue(),32));

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1,32));

    // construct the stride argument
    std::vector<constInt> stride;
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    DenseElementsAttr a1 = co1.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a1.getIntValues())
      stride.push_back(constInt(i.getSExtValue(),32));

    // pad out the stride with -1 to make it 4d
    while (stride.size() < 4)
      stride.push_back(constInt(-1,32));

    APInt offset(32,0);
    if (operands.size() > 3) {
      auto co2 = cast<xilinx::aten::ConstantOp>(operands[3]->getDefiningOp());
      auto ia2 = co2.getAttrOfType<IntegerAttr>("value");
      offset = ia2.getValue();
    }

    ArrayRef<Value*> callops{xVal,
                             shape[0], shape[1], shape[2], shape[3],
                             stride[0], stride[1], stride[2], stride[3],
                             constInt(offset.getSExtValue(), 32)};
;

    FuncOp asstridedFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "as_strided", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(asstridedFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower batchnorm
class BatchNormOpConversion : public ConversionPattern {
public:
  explicit BatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::BatchNormOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle aVal(operands[0]);
    edsc::ValueHandle bVal(operands[1]);
    edsc::ValueHandle cVal(operands[2]);
    edsc::ValueHandle dVal(operands[3]);
    edsc::ValueHandle eVal(operands[4]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[5]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[6]->getDefiningOp());
    auto fa0 = co1.getAttrOfType<FloatAttr>("value");
    APFloat faVal0 = fa0.getValue();

    auto co2 = cast<xilinx::aten::ConstantOp>(operands[7]->getDefiningOp());
    auto fa1 = co2.getAttrOfType<FloatAttr>("value");
    APFloat faVal1 = fa1.getValue();

    auto co3 = cast<xilinx::aten::ConstantOp>(operands[8]->getDefiningOp());
    auto ia1 = co3.getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    auto f32Ty = FloatType::getF32(op->getContext());

    ArrayRef<Value*> callops{aVal, bVal, cVal, dVal, eVal,
                             constInt(iaVal0.getZExtValue(), 1),
                             constFloat(faVal0, f32Ty),
                             constFloat(faVal1, f32Ty),
                             constInt(iaVal1.getZExtValue(), 1)};

    FuncOp batchnormFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "batch_norm", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(batchnormFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower conv2d
class ConvolutionOpConversion : public ConversionPattern {
public:
  explicit ConvolutionOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::ConvolutionOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
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
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
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
 
    ArrayRef<Value*> callops{xVal, wVal, bVal, padCI, kernelCI, strideCI};

    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(convFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower conv2d backward
class ConvolutionBackwardOpConversion : public ConversionPattern {
public:
  explicit ConvolutionBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::ConvolutionBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);

    TensorType result1Ty = op->getResult(1)->getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    TensorType result2Ty = op->getResult(2)->getType().cast<TensorType>();
    Type memRefResult2Ty = mlir::MemRefType::get(result2Ty.getShape(),
                                                 result2Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]); // grad_output
    edsc::ValueHandle arg1(operands[1]); // input
    edsc::ValueHandle arg2(operands[2]); // weight

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
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
 
    ArrayRef<Value*> callops{arg0, arg1, arg2, padCI, kernelCI, strideCI};

    ArrayRef<mlir::Type> retTys{memRefResult0Ty, memRefResult1Ty, memRefResult2Ty};

    FuncOp convFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "conv2d_backward", callops, retTys);

    auto new_call = callOperation(retTys,
                                  rewriter.getSymbolRefAttr(convFunc),
                                  callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// Lower Div
class DivOpConversion : public ConversionPattern {
public:
  explicit DivOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::DivOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);
    edsc::ValueHandle yVal(operands[1]);

    ArrayRef<Value*> callops{xVal, yVal};

    FuncOp divFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "div", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(divFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};
/// Lower LogSoftmax
class LogSoftmaxOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::LogSoftmaxOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle aVal(operands[0]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[1]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    ArrayRef<Value*> callops{aVal,
                             constInt(iaVal0.getSExtValue(), 32),
                             constInt(iaVal1.getZExtValue(), 1)};

    FuncOp logsoftmaxFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "log_softmax", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(logsoftmaxFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower LogSoftmaxBackwardData
class LogSoftmaxBackwardOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::LogSoftmaxBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);
    edsc::ValueHandle arg3(operands[3]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    ArrayRef<Value*> callops{arg0, arg1,
                             constInt(iaVal0.getSExtValue(), 32),
                             arg3};

    FuncOp logsoftmaxBackwardFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                              "log_softmax_backward_data", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(logsoftmaxBackwardFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower maxpool2d
class MaxPoolOpConversion : public ConversionPattern {
public:
  explicit MaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MaxPool2dOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[1], kernel);
    unpack(operands[2], stride);
    unpack(operands[3], pad);

    ArrayRef<Value*> callops{xVal,
                             constInt(kernel[0],32),
                             constInt(stride[0],32),
                             constInt(pad[0],32)};

    FuncOp maxpoolFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                   "max_pool2d", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(maxpoolFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower maxpool2d
class MaxPool2dWithIndicesOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MaxPool2dWithIndicesOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    Type idxTy = op->getResult(1)->getType();
    TensorType tensorIdxTy = idxTy.cast<TensorType>();
    Type memRefIdxTy = mlir::MemRefType::get(tensorIdxTy.getShape(),
                                             tensorIdxTy.getElementType(),
                                             {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
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
    auto co = cast<xilinx::aten::ConstantOp>(operands[5]->getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    ArrayRef<Value*> callops{xVal,
                             constInt(kernel[0],32),
                             constInt(stride[0],32),
                             constInt(pad[0],32),
                             constInt(dilation[0],32),
                             constInt(iaVal.getZExtValue(), 1)};

    ArrayRef<mlir::Type> retTys{memRefResultTy, memRefIdxTy};

    FuncOp maxpoolFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                   "max_pool2d_with_indices", callops, retTys);

    auto new_call = callOperation(retTys,
                         rewriter.getSymbolRefAttr(maxpoolFunc),
                         callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};


/// Lower max_pool2d_with_indicies_backward
class MaxPool2dWithIndicesBackwardOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MaxPool2dWithIndicesBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
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
    auto co = cast<xilinx::aten::ConstantOp>(operands[6]->getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    ArrayRef<Value*> callops{operands[0], operands[1],
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

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// Lower MM
class MMOpConversion : public ConversionPattern {
public:
  explicit MMOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MMOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);
    edsc::ValueHandle yVal(operands[1]);

    ArrayRef<Value*> callops{xVal, yVal};

    FuncOp mmFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                              "mm", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(mmFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower Mul
class MulOpConversion : public ConversionPattern {
public:
  explicit MulOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::MulOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);
    edsc::ValueHandle yVal(operands[1]);

    ArrayRef<Value*> callops{xVal, yVal};

    FuncOp mulFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                               "mul", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(mulFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower batchnorm
class NativeBatchNormOpConversion : public ConversionPattern {
public:
  explicit NativeBatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::NativeBatchNormOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle aVal(operands[0]);
    edsc::ValueHandle bVal(operands[1]);
    edsc::ValueHandle cVal(operands[2]);
    edsc::ValueHandle dVal(operands[3]);
    edsc::ValueHandle eVal(operands[4]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[5]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[6]->getDefiningOp());
    auto fa0 = co1.getAttrOfType<FloatAttr>("value");
    APFloat faVal0 = fa0.getValue();

    auto co2 = cast<xilinx::aten::ConstantOp>(operands[7]->getDefiningOp());
    auto fa1 = co2.getAttrOfType<FloatAttr>("value");
    APFloat faVal1 = fa1.getValue();

    auto f32Ty = FloatType::getF32(op->getContext());

    ArrayRef<Value*> callops{aVal, bVal, cVal, dVal, eVal,
                             constInt(iaVal0.getZExtValue(), 1),
                             constFloat(faVal0, f32Ty),
                             constFloat(faVal1, f32Ty)};

    FuncOp batchnormFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "native_batch_norm", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(batchnormFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// lower NLL Loss backward
class NllLoss2dBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::NllLoss2dBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);
    edsc::ValueHandle arg2(operands[2]);
    edsc::ValueHandle arg3(operands[3]);
    edsc::ValueHandle arg6(operands[6]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[5]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt arg5 = ia1.getValue();

    ArrayRef<Value*> callops{arg0, arg1, arg2, arg3,
                             constInt(arg4.getZExtValue(), 32),
                             constInt(arg5.getZExtValue(), 32),
                             arg6};

    FuncOp nllLoss2dFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                        "nll_loss2d_backward",
                                         callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                                  rewriter.getSymbolRefAttr(nllLoss2dFwdFunc),
                                  callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// lower NLL Loss forward
class NllLoss2dForwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dForwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::NllLoss2dForwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);
    TensorType result1Ty = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);
    edsc::ValueHandle arg2(operands[2]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia1.getValue();

    ArrayRef<Value*> callops{arg0, arg1, arg2,
                             constInt(arg3.getZExtValue(), 32),
                             constInt(arg4.getZExtValue(), 32)};

    ArrayRef<Type> retTy{memRefResult0Ty,memRefResult1Ty};

    FuncOp nllLoss2dFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                        "nll_loss2d_forward",
                                         callops, retTy);

    auto new_call = callOperation(retTy,
                                  rewriter.getSymbolRefAttr(nllLoss2dFwdFunc),
                                  callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// lower NLL Loss backward
class NllLossBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLossBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::NllLossBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType resultTy = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(resultTy.getShape(),
                                                resultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);
    edsc::ValueHandle arg2(operands[2]);
    edsc::ValueHandle arg3(operands[3]);
    edsc::ValueHandle arg6(operands[6]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[5]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt arg5 = ia1.getValue();

    ArrayRef<Value*> callops{arg0, arg1, arg2, arg3,
                             constInt(arg4.getZExtValue(), 32),
                             constInt(arg5.getZExtValue(), 32),
                             arg6};

    FuncOp nllLossFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "nll_loss_backward",
                                       callops, memRefResultTy);

    auto new_call = callOperation(memRefResultTy,
                                  rewriter.getSymbolRefAttr(nllLossFwdFunc),
                                  callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// lower NLL Loss forward
class NllLossForwardOpConversion : public ConversionPattern {
public:
  explicit NllLossForwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::NllLossForwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    TensorType result0Ty = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResult0Ty = mlir::MemRefType::get(result0Ty.getShape(),
                                                 result0Ty.getElementType(),
                                                 {}, 0);
    TensorType result1Ty = op->getResult(0)->getType().cast<TensorType>();
    Type memRefResult1Ty = mlir::MemRefType::get(result1Ty.getShape(),
                                                 result1Ty.getElementType(),
                                                 {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);
    edsc::ValueHandle arg2(operands[2]);

    // reduction
    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt arg3 = ia0.getValue();

    // ignore_index
    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt arg4 = ia1.getValue();

    ArrayRef<Value*> callops{arg0, arg1, arg2,
                             constInt(arg3.getZExtValue(), 32),
                             constInt(arg4.getZExtValue(), 32)};

    ArrayRef<Type> retTy{memRefResult0Ty,memRefResult1Ty};

    FuncOp nllLossFwdFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                      "nll_loss_forward",
                                       callops, retTy);

    auto new_call = callOperation(retTy,
                                  rewriter.getSymbolRefAttr(nllLossFwdFunc),
                                  callops);

    rewriter.replaceOp(op, new_call.getOperation()->getResults());
    return matchSuccess();
  }
};

/// Lower ReLU
class ReLUOpConversion : public ConversionPattern {
public:
  explicit ReLUOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::ReLUOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    ArrayRef<Value*> callops{xVal};

    FuncOp reluFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "relu", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(reluFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower ThresholdBackward
class ThresholdBackwardOpConversion : public ConversionPattern {
public:
  explicit ThresholdBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::ThresholdBackwardOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle arg0(operands[0]);
    edsc::ValueHandle arg1(operands[1]);

    auto co = dyn_cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
    APInt arg2 = ia.getValue();

    ArrayRef<Value*> callops{arg0, arg1,
                             constInt(arg2.getSExtValue(), 32)};

    FuncOp reluFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "threshold_backward",
                                callops,
                                memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(reluFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower transpose
class TransposeOpConversion : public ConversionPattern {
public:
  explicit TransposeOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::TransposeOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    ArrayRef<Value*> callops{xVal};

    FuncOp transposeFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                     "t", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(transposeFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// Lower view
class ViewOpConversion : public ConversionPattern {
public:
  explicit ViewOpConversion(MLIRContext *context)
      : ConversionPattern(xilinx::aten::ViewOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    Type resultTy = op->getResult(0)->getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    Type memRefResultTy = mlir::MemRefType::get(tensorResultTy.getShape(),
                                                tensorResultTy.getElementType(),
                                                {}, 0);

    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    edsc::ValueHandle xVal(operands[0]);

    // construct the shape argument
    std::vector<constInt> shape;
    auto co = cast<xilinx::aten::ConstantOp>(operands[1]->getDefiningOp());
    DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a.getIntValues())
      shape.push_back(constInt(i.getSExtValue(),32));

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1,32));

    ArrayRef<Value*> callops{xVal, shape[0], shape[1], shape[2], shape[3]};

    FuncOp viewFunc = getATenFn(op->getParentOfType<ModuleOp>(),
                                "view", callops, memRefResultTy);

    auto new_call = call(memRefResultTy,
                         rewriter.getSymbolRefAttr(viewFunc),
                         callops);

    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }
};

/// This is the main class registering our individual converter classes with
/// the DialectConversion framework in MLIR.
class ATenTypeConverter : public TypeConverter {
protected:
  /// Convert an ATen type, this gets called for block and region arguments, and
  /// attributes.
  Type convertType(Type t) override {
    if (auto tensor = t.dyn_cast<TensorType>())
      return mlir::MemRefType::get(tensor.getShape(), tensor.getElementType(), {}, 0);
    return t;
  }
};


struct ATenLoweringPass : public ModulePass<ATenLoweringPass> {

  LogicalResult convertToLLVM(ModuleOp module) {
    LLVMTypeConverter converter(module.getContext());
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(converter, patterns);

    ConversionTarget target(*module.getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyFullConversion(module, target, patterns, &converter)))
      return failure();

    return success();
  }

  void runOnModule() override {
    ATenTypeConverter typeConverter;
    OwningRewritePatternList atenPatterns;

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
                        LogSoftmaxBackwardOpConversion, DivOpConversion>(
        &getContext());

    mlir::populateFuncOpTypeConversionPattern(atenPatterns,
                                              &getContext(),
                                              typeConverter);

    // tablegen patterns
    populateATenToStdPatterns(&getContext(), atenPatterns);

    // Perform aten specific lowering.
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineOpsDialect, LLVM::LLVMDialect, StandardOpsDialect>();
    target.addLegalOp<xilinx::aten::AllocOp, xilinx::aten::TypeCastOp>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyPartialConversion(getModule(), target, atenPatterns,
                                      &typeConverter))) {
      emitError(UnknownLoc::get(getModule().getContext()),
                "error lowering ATen\n");
      signalPassFailure();
    }

    // remove dead constant ops
    for (auto function : getModule().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        auto constOp = dyn_cast<xilinx::aten::ConstantOp>(op);
        if (!constOp)
          return;
        if (op->use_empty())
          op->erase();
      });
    }

    //convertToLLVM(getModule());
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
