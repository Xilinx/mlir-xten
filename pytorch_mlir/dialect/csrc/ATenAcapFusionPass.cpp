// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "ATenDialect.h"
#include "ATenAcapFusionPass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/AffineOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/EDSC/Builders.h"
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
using namespace xilinx;

namespace {

#include "ATenAcapFusion.cpp.inc"

struct ATenAcapFusionPass : public ModulePass<ATenAcapFusionPass> {

  void runOnModule() override {
    auto module = getModule();
    auto context = module.getContext();
 
    LLVMTypeConverter typeConverter(context);

    // tablegen patterns
    OwningRewritePatternList fusionPatterns;
    populateWithGenerated(context, &fusionPatterns);

    // Perform aten specific Fusion.
    ConversionTarget target(*context);

    target.addLegalDialect<AffineOpsDialect, LLVM::LLVMDialect,
                           StandardOpsDialect, loop::LoopOpsDialect>();

    target.addLegalOp<xilinx::aten::AcapConv2dBatchNormReLUOp>();
    target.addLegalOp<xilinx::aten::AcapConv2dReLUOp>();
    target.addLegalOp<xilinx::aten::AcapNoOp>();

    if (failed(applyPartialConversion(module, target, fusionPatterns, &typeConverter))) {
      emitError(UnknownLoc::get(context), "error fusing ATen\n");
      signalPassFailure();
      assert(0);
    }

  }
};

}// namespace


namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenAcapFusionPass() {
  return std::make_unique<ATenAcapFusionPass>();
}

} // namespace aten
} // namespace xilinx
