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
#include "mlir/IR/Operation.h"
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
#include <map>
#include <sstream>

#define DEBUG_TYPE "aten-to-xten-pass"

using namespace mlir;
using namespace xilinx;
using namespace mlir::torch;

namespace {

// Get ancestor ops based on operands of an mlir operation
SmallVector<Operation*> getPreviousOps(mlir::Operation* op) {
  SmallVector<Operation*> ancestors;
  for (Value operand : op->getOperands()) {
    if (Operation *ancestor = operand.getDefiningOp()) {
      ancestors.push_back(ancestor);
    }
  }
  return ancestors;
}

// Adapter fun because tablegen can only bind result and not the op itself.
bool fuseFirstC2dInTensorAdd(OpResult left, OpResult right) {

  // Create map to find common ancestor node
  SmallVector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  // Get left and right operation
  Operation *leftOp = left.getOwner();
  Operation *rightOp = right.getOwner();

  // Initialize starting operation based on left op
  worklist.push_back(leftOp);
  worklistMap.insert(std::pair<Operation*, unsigned>(leftOp, 0));

  // Iterate backwards over left op branch
  while (!worklist.empty()) {
    // Get next worklist element
    Operation* nextOp = *(worklist.begin());
    unsigned nextLen = worklistMap[nextOp];

    // Get ancestor ops and push to worklist
    SmallVector<Operation*> prevOps = getPreviousOps(nextOp);
    for (Operation* op : prevOps) {
      worklist.push_back(op);
      worklistMap.insert(std::pair<Operation*, unsigned>(op, nextLen + 1));
    }
    worklist.erase (worklist.begin(), worklist.begin() + 1);
  }

  // Iterate backwards over right op branch
  worklist.push_back(rightOp);
  worklistMap.insert(std::pair<Operation*, unsigned>(rightOp, 0));

  while (!worklist.empty()) {
    Operation* nextOp = *(worklist.begin());
    unsigned nextLen = worklistMap[nextOp];
    SmallVector<Operation*> prevOps = getPreviousOps(nextOp);

    for (Operation* op : prevOps) {
      if (worklistMap.find(op) != worklistMap.end()) {

        // TODO: Ignore or classify ops to avoid ancestors that are not ml ops

        if (worklistMap[op] < nextLen + 1) {
          // Left branch is the skip branch. Fuse right convolution op with
          // element-wise add.
          return false;
        }
        return true;
      }
      worklist.push_back(op);
      worklistMap.insert(std::pair<Operation*, unsigned>(op, nextLen + 1));
    }
  }

  // No skip connection found. Assuming right branch is skip connection.
  // Fuse left convolution op with element-wise add.
  return true;
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
