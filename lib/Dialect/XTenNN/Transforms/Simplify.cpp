//===- Simplify.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Implements the XTenNN simplification pass.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Dialect/XTenNN/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xtennn-simplify"

namespace amd::xten_nn {
using namespace mlir;
#define GEN_PASS_DEF_XTENNNSIMPLIFY
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"
} // namespace amd::xten_nn

using namespace llvm;
using namespace mlir;
using namespace amd::xten_nn;

namespace {

/// Removes unused capture arguments from EnclaveOp ops.
class RemoveUnusedCaptures : public OpInterfaceRewritePattern<EnclaveOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(EnclaveOp op,
                                PatternRewriter &rewriter) const override {
    // Collect all unused block arguments.
    auto unused = to_vector(
        make_filter_range(op.getEnclaveBody().getArguments(),
                          [](BlockArgument arg) { return arg.use_empty(); }));

    if (unused.empty())
      return failure();

    // Update the op in-place.
    rewriter.modifyOpInPlace(op, [&]() { op.uncapture(unused); });
    return success();
  }
};

/// Removes unused return values from EnclaveOp ops.
class RemoveUnusedReturns : public OpInterfaceRewritePattern<EnclaveOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(EnclaveOp op,
                                PatternRewriter &rewriter) const override {
    // There must be results we can remove.
    if (!any_of(op->getResults(),
                [](OpResult result) { return result.use_empty(); }))
      return failure();

    // NOTE: Results / Regions cannot erased / transferred in-place.
    // Clone the op
    OperationState state(op.getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addAttributes(op->getAttrs());
    for (auto &region : op->getRegions()) {
      IRMapping map;
      region.cloneInto(state.addRegion(), map);
    }

    // Build the new result list, remembering indices that were deleted.
    SmallVector<unsigned> indices;
    for (auto result : op->getResults())
      if (!result.use_empty())
        state.types.push_back(result.getType());
      else
        indices.push_back(result.getResultNumber());

    // Create the new op, and erase the deleted results from the terminator.
    rewriter.setInsertionPointAfter(op);
    auto newOp = cast<EnclaveOp>(rewriter.create(state));
    auto newTerminator = newOp.getTerminator();
    for (unsigned idx : reverse(indices))
      newTerminator->eraseOperand(idx);

    // Replace the uses of the old op with the new op's results.
    unsigned newIndex = 0;
    for (auto result : op->getResults()) {
      if (result.use_empty())
        continue;
      result.replaceAllUsesWith(newOp->getResult(newIndex++));
    }

    // The op was updated out-of-place.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void amd::xten_nn::populateSimplifyPatterns(RewritePatternSet &patterns) {
  patterns.add<RemoveUnusedCaptures, RemoveUnusedReturns>(
      patterns.getContext());
}

namespace {

/// Simplifies DLNN networks.
class SimplifyPass
    : public amd::xten_nn::impl::XTenNNSimplifyBase<SimplifyPass> {
public:
  using XTenNNSimplifyBase::XTenNNSimplifyBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    populateSimplifyPatterns(patterns);

    std::ignore = applyPatternsAndFoldGreedily(
        getOperation(), FrozenRewritePatternSet(std::move(patterns)));
  }
};

} // namespace

std::unique_ptr<Pass> amd::xten_nn::createSimplifyPass() {
  return std::make_unique<SimplifyPass>();
}
