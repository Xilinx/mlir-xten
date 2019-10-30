
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

using namespace mlir;

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
    MemRefView vRes(result), vLHS(lhs), vRHS(rhs);
    IndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
    IndexHandle i, j, k, M(vRes.ub(0));
    if (vRes.rank() == 1) {
      LoopNestBuilder({&i}, {zero}, {M},
                      {1})([&] { iRes(i) = iLHS(i) + iRHS(i); });
    } else if (vRes.rank() == 2) {
      IndexHandle N(vRes.ub(1));
      LoopNestBuilder({&i, &j}, {zero, zero}, {M, N},
                      {1, 1})([&] { iRes(i, j) = iLHS(i, j) + iRHS(i, j); });
    } else {
        assert(vRes.rank() == 3 && "only ranks <= 3 are supported right now");
        IndexHandle N(vRes.ub(1));
        IndexHandle O(vRes.ub(2));
        
      LoopNestBuilder({&i, &j, &k}, {zero, zero, zero}, {M, N, O},
                      {1, 1, 1})([&] { iRes(i, j, k) = iLHS(i, j, k) + iRHS(i, j, k); });
    }

    // Return the newly allocated buffer, with a type.cast to preserve the
    // consumers.
    rewriter.replaceOp(op, {typeCast(rewriter, result, add.getType())});
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
    FuncOp convFunc = getConvolution2dForward(op->getParentOfType<ModuleOp>(),
                                              operands,
                                              memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;
    using constInt = intrinsics::constant_int;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle xVal(operands[0]);
    ValueHandle wVal(operands[1]);
    ValueHandle bVal(operands[2]);

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

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(convFunc),
                         {xVal, wVal, bVal,
                          constInt(pad[0],32),
                          constInt(kernel[0],32),
                          constInt(stride[0],32)});
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getConvolution2dForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto convFunc = module.lookupSymbol<FuncOp>("conv2d_forward");
    if (convFunc)
      return convFunc;

    Builder builder(module);

    auto i32Ty = mlir::IntegerType::get(32, module.getContext());
    auto convFuncTy = builder.getFunctionType({operands[0]->getType(),
                                               operands[1]->getType(),
                                               operands[2]->getType(),
                                               i32Ty, i32Ty, i32Ty},
                                              {retTy});
    convFunc = FuncOp::create(builder.getUnknownLoc(), "conv2d_forward", convFuncTy);
    module.push_back(convFunc);
    return convFunc;
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
    FuncOp batchnormFunc = getBatchNorm2dForward(op->getParentOfType<ModuleOp>(),
                                                 operands,
                                                 memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;
    using constInt = intrinsics::constant_int;
    using constFloat = intrinsics::constant_float;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle aVal(operands[0]);
    ValueHandle bVal(operands[1]);
    ValueHandle cVal(operands[2]);
    ValueHandle dVal(operands[3]);
    ValueHandle eVal(operands[4]);

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

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(batchnormFunc),
                         {aVal, bVal, cVal, dVal, eVal,
                          constInt(iaVal0.getZExtValue(), 1),
                          constFloat(faVal0, f32Ty),
                          constFloat(faVal1, f32Ty),
                          constInt(iaVal1.getZExtValue(), 1)
                          });
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getBatchNorm2dForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto batchnormFunc = module.lookupSymbol<FuncOp>("batch_norm_forward");
    if (batchnormFunc)
      return batchnormFunc;

    Builder builder(module);

    auto boolTy = mlir::IntegerType::get(1, module.getContext());
    auto f32Ty = FloatType::getF32(module.getContext());
    auto batchnormFuncTy = builder.getFunctionType({operands[0]->getType(),
                                                    operands[1]->getType(),
                                                    operands[2]->getType(),
                                                    operands[3]->getType(),
                                                    operands[4]->getType(),
                                                    boolTy, f32Ty, f32Ty, boolTy},
                                                   {retTy});
    batchnormFunc = FuncOp::create(builder.getUnknownLoc(), "batch_norm_forward", batchnormFuncTy);
    module.push_back(batchnormFunc);
    return batchnormFunc;
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
    FuncOp maxpoolFunc = getMaxPool2dForward(op->getParentOfType<ModuleOp>(),
                                             operands,
                                             memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;
    using constInt = intrinsics::constant_int;
    using constFloat = intrinsics::constant_float;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle xVal(operands[0]);

    auto unpack = [](auto &op, auto &v) -> void {
      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
      for (auto i : a.getIntValues())
        v.push_back(i.getSExtValue());
    };

    std::vector<uint64_t> pad, kernel, stride;
    unpack(operands[1], pad);
    unpack(operands[2], kernel);
    unpack(operands[3], stride);

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(maxpoolFunc),
                         {xVal,
                          constInt(pad[0],32),
                          constInt(kernel[0],32),
                          constInt(stride[0],32)
                          });
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getMaxPool2dForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto maxpoolFunc = module.lookupSymbol<FuncOp>("maxpool_forward");
    if (maxpoolFunc)
      return maxpoolFunc;

    Builder builder(module);

    auto i32Ty = mlir::IntegerType::get(32, module.getContext());
    auto maxpoolFuncTy = builder.getFunctionType({operands[0]->getType(),
                                                  i32Ty, i32Ty, i32Ty},
                                                 {retTy});
    maxpoolFunc = FuncOp::create(builder.getUnknownLoc(), "maxpool_forward", maxpoolFuncTy);
    module.push_back(maxpoolFunc);
    return maxpoolFunc;
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
    FuncOp addFunc = getAddForward(op->getParentOfType<ModuleOp>(),
                                     operands,
                                     memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;
    using constInt = intrinsics::constant_int;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle xVal(operands[0]);
    ValueHandle yVal(operands[1]);

    auto co = cast<xilinx::aten::ConstantOp>(operands[2]->getDefiningOp());
    auto ia = co.getAttrOfType<IntegerAttr>("value");
    APInt iaVal = ia.getValue();

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(addFunc),
                         {xVal, yVal, constInt(iaVal.getSExtValue(), 32)});
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getAddForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto addFunc = module.lookupSymbol<FuncOp>("add_forward");
    if (addFunc)
      return addFunc;

    Builder builder(module);

    auto addFuncTy = builder.getFunctionType({operands[0]->getType(),
                                              operands[1]->getType(),
                                              operands[2]->getType()},
                                              {retTy});
    addFunc = FuncOp::create(builder.getUnknownLoc(), "add_forward", addFuncTy);
    module.push_back(addFunc);
    return addFunc;
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
    FuncOp addmmFunc = getAddmmForward(op->getParentOfType<ModuleOp>(),
                                     operands,
                                     memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;
    using constInt = intrinsics::constant_int;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle aVal(operands[0]);
    ValueHandle bVal(operands[1]);
    ValueHandle cVal(operands[2]);

    auto co0 = cast<xilinx::aten::ConstantOp>(operands[3]->getDefiningOp());
    auto ia0 = co0.getAttrOfType<IntegerAttr>("value");
    APInt iaVal0 = ia0.getValue();

    auto co1 = cast<xilinx::aten::ConstantOp>(operands[4]->getDefiningOp());
    auto ia1 = co1.getAttrOfType<IntegerAttr>("value");
    APInt iaVal1 = ia1.getValue();

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(addmmFunc),
                         {aVal, bVal, cVal,
                          constInt(iaVal0.getSExtValue(), 32),
                          constInt(iaVal1.getSExtValue(), 32),
                          });
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getAddmmForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto addmmFunc = module.lookupSymbol<FuncOp>("addmm_forward");
    if (addmmFunc)
      return addmmFunc;

    Builder builder(module);

    auto i32Ty = mlir::IntegerType::get(32, module.getContext());
    auto addmmFuncTy = builder.getFunctionType({operands[0]->getType(),
                                              operands[1]->getType(),
                                              operands[2]->getType(),
                                              i32Ty, i32Ty},
                                              {retTy});
    addmmFunc = FuncOp::create(builder.getUnknownLoc(), "addmm_forward", addmmFuncTy);
    module.push_back(addmmFunc);
    return addmmFunc;
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
    FuncOp reluFunc = getReLUForward(op->getParentOfType<ModuleOp>(),
                                     operands,
                                     memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle xVal(operands[0]);

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(reluFunc),
                         {xVal});
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getReLUForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto reluFunc = module.lookupSymbol<FuncOp>("relu_forward");
    if (reluFunc)
      return reluFunc;

    Builder builder(module);

    auto reluFuncTy = builder.getFunctionType({operands[0]->getType()},
                                              {retTy});
    reluFunc = FuncOp::create(builder.getUnknownLoc(), "relu_forward", reluFuncTy);
    module.push_back(reluFunc);
    return reluFunc;
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
    FuncOp transposeFunc = getTransposeForward(op->getParentOfType<ModuleOp>(),
                                     operands,
                                     memRefResultTy);

    using namespace edsc;
    using call = intrinsics::ValueBuilder<mlir::CallOp>;

    auto loc = op->getLoc();
    ScopedContext scope(rewriter, loc);

    ValueHandle xVal(operands[0]);

    auto new_call = call(memRefResultTy, rewriter.getSymbolRefAttr(transposeFunc),
                         {xVal});
    rewriter.replaceOp(op, {new_call});
    return matchSuccess();
  }

  FuncOp getTransposeForward(ModuleOp module, ArrayRef<Value *> &operands, Type &retTy) const
  {
    auto transposeFunc = module.lookupSymbol<FuncOp>("transpose_forward");
    if (transposeFunc)
      return transposeFunc;

    Builder builder(module);

    auto transposeFuncTy = builder.getFunctionType({operands[0]->getType()},
                                              {retTy});
    transposeFunc = FuncOp::create(builder.getUnknownLoc(), "transpose_forward", transposeFuncTy);
    module.push_back(transposeFunc);
    return transposeFunc;
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
                        BatchNormOpConversion, MaxPoolOpConversion,
                        AddmmOpConversion>(
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
  }
} // namespace
