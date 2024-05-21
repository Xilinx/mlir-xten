//===- QDQConcat.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Declares rewrites for QDQ-Concat patterns.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Dialect/XTenNN/Transforms/CanonicalizePass.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::tosa;
using namespace amd::xten_nn;

namespace {
struct RemoveQDQBetweenConcat : public OpRewritePattern<DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter &rewriter) const override {
    // Match concat->QDQ->concat and remove QDQ, if concats would be foldable.
    // Removing a QDQ is already destructive. Try to be a little-less
    // destructive by checking that the QDQ nodes have the same shift.
    auto quantize =
        llvm::dyn_cast_or_null<QuantizeOp>(op.getInput().getDefiningOp());
    if (!quantize) {
      return rewriter.notifyMatchFailure(
          op, "DequantizeOp input not produced by QuantizeOp.");
    }

    if (quantize.getShift() != op.getShift()) {
      return rewriter.notifyMatchFailure(
          op, "DequantizeOp and QuantizeOp do not share the same shift value.");
    }

    // Try to match an incoming concat
    auto producer =
        dyn_cast_or_null<ConcatOp>(quantize.getInput().getDefiningOp());
    if (producer == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "QDQ input not produced by TOSA concat.");
    }

    uint64_t axis = producer.getAxis();

    bool modified = false;
    for (Operation *user : op->getUsers()) {
      auto userConcat = dyn_cast<ConcatOp>(user);
      if (userConcat == nullptr)
        continue;

      // Concats are foldable if they concatenate on the same axis
      if (userConcat.getAxis() != axis)
        continue;

      // Foldable concats found. Folding the concats is the responsibility of
      // TOSA canonicalization. Rewire the concat->QDQ->concat to be
      // concat->concat instead.
      rewriter.startOpModification(user);
      user->replaceUsesOfWith(op, producer.getOutput());
      rewriter.finalizeOpModification(user);
      modified = true;
    }

    if (modified)
      return success();

    return rewriter.notifyMatchFailure(
        op, "No foldable concats found around QDQ node.");
  }
};
} // namespace

void amd::xten_nn::populateQDQConcatPatterns(RewritePatternSet &patterns) {
  patterns.add<RemoveQDQBetweenConcat>(patterns.getContext());
}
