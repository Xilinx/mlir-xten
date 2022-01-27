//===- ATenToXTenPass.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "xten/Conversion/ATenToXTenPass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
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

#define DEBUG_TYPE "aten-to-xten-pass"

using namespace mlir;
using namespace xilinx::xten;

namespace {

#include "xten/Conversion/ATenToXTen.cpp.inc"

struct ATenToXTenPass : public ATenToXTenBase<ATenToXTenPass> {

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
     registry.insert<xilinx::xten::XTenDialect>();
     registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();
    
    // tablegen patterns
    RewritePatternSet fusionPatterns(&getContext());
    populateWithGenerated(fusionPatterns);

    // Perform aten specific Fusion.
    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           StandardOpsDialect, scf::SCFDialect>();

    target.addLegalOp<xilinx::xten::Conv2dBatchNormReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dLReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dLReLUMaxPoolOp>();
    target.addLegalOp<xilinx::xten::Conv2dOp>();
    target.addLegalOp<xilinx::xten::NoOp>();
    if (failed(applyPatternsAndFoldGreedily(module, /*target,*/ std::move(fusionPatterns)))) {
      emitError(UnknownLoc::get(context), "error translating or fusing ATen to XTen\n");
      signalPassFailure();
      assert(0);
    }

  }
};

}// namespace


namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>>
createATenToXTenPass() {
  return std::make_unique<ATenToXTenPass>();
}

} // namespace xten
} // namespace xilinx
