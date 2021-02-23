// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"
#include "ATenToAIRPass.h"

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

#define DEBUG_TYPE "aten-to-air-pass"

using namespace mlir;
using namespace xilinx;

namespace {

#include "ATenToAIR.cpp.inc"

struct ATenToAIRPass : public PassWrapper<ATenToAIRPass,
                                          OperationPass<ModuleOp>> {

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<xilinx::air::airDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();
    
    // tablegen patterns
    OwningRewritePatternList fusionPatterns;
    populateWithGenerated(context, fusionPatterns);

    // Perform aten specific Fusion.
    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           StandardOpsDialect, scf::SCFDialect>();

    target.addLegalOp<xilinx::air::Conv2dBatchNormReLUOp>();
    target.addLegalOp<xilinx::air::Conv2dReLUOp>();
    target.addLegalOp<xilinx::air::Conv2dOp>();
    target.addLegalOp<xilinx::air::NoOp>();
    if (failed(applyPatternsAndFoldGreedily(module, /*target,*/ std::move(fusionPatterns)))) {
      emitError(UnknownLoc::get(context), "error fusing ATen\n");
      signalPassFailure();
      assert(0);
    }

  }
};

}// namespace


namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenToAIRPass() {
  return std::make_unique<ATenToAIRPass>();
}

} // namespace aten
} // namespace xilinx
