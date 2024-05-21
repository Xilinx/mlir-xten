//===- CanonicalizePass.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Defines the XTenNN canonicalize pass.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/Transforms/CanonicalizePass.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "xten-canonicalize"

namespace amd::xten_nn {
using namespace mlir;
#define GEN_PASS_DEF_XTENNNCANONICALIZEPASS
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"
} // namespace amd::xten_nn

using namespace mlir;
using namespace amd::xten_nn;

namespace {
struct CanonicalizePass
    : public amd::xten_nn::impl::XTenNNCanonicalizePassBase<CanonicalizePass> {
  using XTenNNCanonicalizePassBase::XTenNNCanonicalizePassBase;

  CanonicalizePass(bool allowDestructivePatterns) {
    this->allowDestructivePatterns = allowDestructivePatterns;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    populateCanonicalizePatterns(patterns);
    if (allowDestructivePatterns)
      populateDestructiveCanonicalizePatterns(patterns);

    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed())
      signalPassFailure();
  }
};
} // namespace

void amd::xten_nn::populateCanonicalizePatterns(
    RewritePatternSet & /*patterns*/) {
  // TODO: FXML-2034 add non-destructive patterns
}

void amd::xten_nn::populateDestructiveCanonicalizePatterns(
    RewritePatternSet &patterns) {
  // Add destructive canonicalization patterns here
  populateQDQConcatPatterns(patterns);
}

std::unique_ptr<Pass> amd::xten_nn::createCanonicalizePass() {
  return std::make_unique<CanonicalizePass>();
}

std::unique_ptr<Pass>
amd::xten_nn::createCanonicalizePass(bool allowDestructivePatterns) {
  return std::make_unique<CanonicalizePass>(allowDestructivePatterns);
}
