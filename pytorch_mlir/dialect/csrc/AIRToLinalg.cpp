// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "AIRToLinalgPass.h"

#include "AIRDialect.h"

// #include "Util.h"

// #include "mlir/Analysis/Utils.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/EDSC/Builders.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/Dialect/StandardOps/EDSC/Builders.h"
// #include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
// #include "mlir/Dialect/SCF/EDSC/Builders.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/EDSC/Builders.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/IntegerSet.h"
// #include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/Support/MathExtras.h"
// #include "mlir/Transforms/LoopUtils.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-to-affine-pass"

using namespace mlir;
using namespace xilinx;

namespace {

#include "AIRToLinalg.cpp.inc"

class AIRMMOpConversion : public ConversionPattern {
public:
  explicit AIRMMOpConversion(MLIRContext *context)
      : ConversionPattern(air::MMOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    //auto mmult = cast<air::MMOp>(op);
    //auto loc = mmult.getLoc();

    return success();
  }
};


class AIRToLinalgPass : public PassWrapper<AIRToLinalgPass,
                                           OperationPass<ModuleOp>> {

public:
  AIRToLinalgPass() {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    OwningRewritePatternList patterns;
    populateWithGenerated(context, patterns);

    //patterns.insert<AIRMMOpConversion>(context);

    // populateFuncOpTypeConversionPattern(patterns,
    //                                     context,
    //                                     typeConverter);

    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, linalg::LinalgDialect,
                           StandardOpsDialect, scf::SCFDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
       return typeConverter.isSignatureLegal(op.getType());
    });


    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering AIR to Linalg\n");
      signalPassFailure();
      //assert(0);
    }
  }

private:

};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRToLinalgPass() {
  return std::make_unique<AIRToLinalgPass>();
}

} // namespace air
} // namespace xilinx

void xilinx::air::registerAIRToLinalgPass() {
    PassRegistration<AIRToLinalgPass>(
      "air-to-linalg",
      "Lower AIR dialect to Linalg dialect");
}
