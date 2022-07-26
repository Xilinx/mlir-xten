//===- ATenToXTenPass.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 - 2022 Xilinx Inc.
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
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

// Get preceding useful ops based on operands of an mlir operation
SmallVector<Operation *> getUsefulInputOps(mlir::Operation *op) {
  SmallVector<Operation *> ancestors;
  for (Value operand : op->getOperands()) {
    if (Operation *ancestor = operand.getDefiningOp()) {
      ancestors.push_back(ancestor);
    }
  }

  // Ops without operands and list construct ops can be ignored
  llvm::erase_if(ancestors, [](Operation *op) {
    return (op->getOperands().empty() ||
            mlir::isa<Torch::PrimListConstructOp>(op));
  });
  return ancestors;
}

// Tablegen adapter to check which op to fuse
//
// Traverse graph backwards and return true if the left op has a longer
// branch than the right op. The length of the op branches is determined
// by the first common ancestor.
bool isLongestBranch(OpResult left, OpResult right) {

  // Create map to find common ancestor node
  SmallVector<Operation *> worklist;
  DenseMap<Operation *, unsigned> opDistanceMap;

  // Get left and right operation
  Operation *leftOp = left.getOwner();
  Operation *rightOp = right.getOwner();

  // Ignore case where both ops are the same
  if (leftOp == rightOp)
    return true;

  // Initialize starting operation based on left op
  worklist.push_back(leftOp);
  opDistanceMap.insert(std::pair<Operation *, unsigned>(leftOp, 0));

  // Iterate backwards over left op branch
  while (!worklist.empty()) {
    auto curOp = opDistanceMap.find(worklist.front());
    SmallVector<Operation *> inputOps = getUsefulInputOps(curOp->first);
    unsigned nextDistance = curOp->second + 1;

    for (Operation *inputOp : inputOps) {
      auto inputOpDistancePair = opDistanceMap.find(inputOp);
      if (inputOpDistancePair == opDistanceMap.end()) {
        worklist.push_back(inputOp);
        opDistanceMap.insert({inputOp, nextDistance});
      } else if (inputOpDistancePair->second <
                 nextDistance) { // Always store longest path
        inputOpDistancePair->second = nextDistance;
      }
    }
    worklist.erase(worklist.begin());
  }

  // Iterate backwards over right op branch
  worklist.push_back(rightOp);
  opDistanceMap.insert(std::pair<Operation *, unsigned>(rightOp, 0));

  while (!worklist.empty()) {
    auto curOp = opDistanceMap.find(worklist.front());
    SmallVector<Operation *> inputOps = getUsefulInputOps(curOp->first);
    unsigned nextDistance = curOp->second + 1;

    for (Operation *inputOp : inputOps) {
      auto inputOpDistancePair = opDistanceMap.find(inputOp);

      if ((inputOpDistancePair != opDistanceMap.end()) &&
          (nextDistance >= inputOpDistancePair->second)) {
        // Left branch is the skip branch. Fuse right convolution op with
        // element-wise add.
        return false;
      } else {
        // Continue searching since we want to find the longest branch in
        // multi-branch scenarios
        worklist.push_back(inputOp);
        opDistanceMap.insert(
            std::pair<Operation *, unsigned>(inputOp, nextDistance));
      }
    }
    worklist.erase(worklist.begin());
  }

  // No skip connection found. Assuming right branch is skip connection.
  // Fuse left convolution op with element-wise add.
  return true;
}

// Tablegen adapter to check if attributes allow a conversion from `ReduceMean`
// to `GlobalAveragePool`.
bool isReduceMeanGlobalAveragePool2D(Value dims, Value keepdims) {

  Torch::PrimListConstructOp dimsOp =
      llvm::dyn_cast<Torch::PrimListConstructOp>(dims.getDefiningOp());
  Torch::ConstantBoolOp keepdimsOp =
      llvm::dyn_cast<Torch::ConstantBoolOp>(keepdims.getDefiningOp());

  if (!dimsOp || !keepdimsOp)
    return false;

  SmallVector<APInt> axes;
  for (Value operand : dimsOp.getOperands()) {
    if (auto axesValue =
            llvm::dyn_cast<Torch::ConstantIntOp>(operand.getDefiningOp()))
      axes.push_back(axesValue.value());
  }

  // Check that arguments match expectations for global average pool
  if (keepdimsOp.value() == 1 && axes.size() == 2 && axes[0] == 2 &&
      axes[1] == 3)
    return true;

  return false;
}

// Tablegen adapter to check if attributes allow a conversion from `AveragePool`
// to `GlobalAveragePool`.
bool isAdaptiveAvgPoolGlobalAveragePool2D(Value outsizes) {
  Torch::PrimListConstructOp outsizesOp =
      llvm::dyn_cast<Torch::PrimListConstructOp>(outsizes.getDefiningOp());

  SmallVector<APInt> dims;
  for (Value operand : outsizesOp.getOperands()) {
    if (auto axesValue =
            llvm::dyn_cast<Torch::ConstantIntOp>(operand.getDefiningOp()))
      dims.push_back(axesValue.value());
  }

  // Check that arguments match expectations for global average pool
  if (dims.size() == 2 && dims[0] == 1 && dims[1] == 1)
    return true;

  return false;
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

    target.addLegalDialect<AffineDialect, LLVM::LLVMDialect, func::FuncDialect,
                           scf::SCFDialect>();

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
