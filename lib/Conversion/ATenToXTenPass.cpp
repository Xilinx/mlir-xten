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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
//#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

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
using namespace xilinx;
using namespace mlir::torch;

namespace {

long getInputSize(xten::Conv2dOp &c2d) {
  TensorType inputType = c2d.input().getType().dyn_cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (inputType == nullptr || !inputType.hasRank())
    return 0;
  auto shape = inputType.getShape();
  // multiply all dimensions together
  long result = 1;
  for (auto i: shape)
    result *= i;
  return result;
}

// Pick conv2d with smallest input size, because we expect that this one will need less L1 storage
// The goal is to minimize memory transfers.
// This might not be the best implementation, may be
// improved later when more low-level details are known.
bool fuseFirstC2dInTensorAddImpl(xten::Conv2dOp &c2d0, xten::Conv2dOp &c2d1) {
  auto s0 = getInputSize(c2d0);
  auto s1 = getInputSize(c2d1);  
  return s0 && s1 && s0 < s1;
}
// adapter fun because tablegen can only bind result and not the op itself.
bool fuseFirstC2dInTensorAdd(OpResult a, OpResult b) {
  auto c2d0 = cast<xten::Conv2dOp>(a.getOwner());
  auto c2d1 = cast<xten::Conv2dOp>(b.getOwner());
  return fuseFirstC2dInTensorAddImpl(c2d0, c2d1);
}

namespace atenToXten {
#include "xten/Conversion/ATenToXTen.cpp.inc"
}

namespace xtenToXtenCleanup {
#include "xten/Conversion/XTenFusions.cpp.inc"
}


struct ATenToXTenPass : public xten::ATenToXTenBase<ATenToXTenPass> {

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::xten::XTenDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    // tablegen patterns
    RewritePatternSet fusionPatterns(&getContext());
    atenToXten::populateWithGenerated(fusionPatterns);

    // Perform aten specific Fusion.
    ConversionTarget target(*context);

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect,
                           func::FuncDialect, scf::SCFDialect>();

    target.addLegalOp<xilinx::xten::Conv2dBatchNormReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dLReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dLReLUMaxPoolOp>();
    target.addLegalOp<xilinx::xten::Conv2dOp>();
    target.addLegalOp<xilinx::xten::Conv2dTensorAddOp>();
    target.addLegalOp<xilinx::xten::Conv2dTensorAddReLUOp>();
    target.addLegalOp<xilinx::xten::Conv2dTensorAddLReLUOp>();
    target.addLegalOp<xilinx::xten::NoOp>();
    if (failed(applyPatternsAndFoldGreedily(
            module, /*target,*/ std::move(fusionPatterns)))) {
      emitError(UnknownLoc::get(context),
                "error translating or fusing ATen to XTen\n");
      signalPassFailure();
      assert(0);
    }

    RewritePatternSet cleanupPatterns(&getContext());
    xtenToXtenCleanup::populateWithGenerated(cleanupPatterns);

    if (failed(applyPatternsAndFoldGreedily(
            module, /*target,*/ std::move(cleanupPatterns)))) {
      emitError(UnknownLoc::get(context),
                "error translating or fusing ATen to XTen\n");
      signalPassFailure();
      assert(0);
    }


  }
};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createATenToXTenPass() {
  return std::make_unique<ATenToXTenPass>();
}

} // namespace xten
} // namespace xilinx
