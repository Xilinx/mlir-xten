#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Transform/Passes.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace mlir::tosa;
using namespace amd::xten_nn;

namespace {
struct XTenQDQConcat : public OpRewritePattern<DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter &rewriter) const override {
    // Match concat->QDQ->concat and remove QDQ, if concats would be foldable.
    // Try to match an incoming concat
    Value val;
    if (!matchPattern(op.getInput(), m_Op<QuantizeOp>(matchers::m_Any(&val)))) {
      return rewriter.notifyMatchFailure(
          op, "Dequantize input not produced by Quantize.");
    }

    auto producer = dyn_cast_or_null<ConcatOp>(val.getDefiningOp());
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
      rewriter.startRootUpdate(user);
      user->replaceUsesOfWith(op, producer.getOutput());
      rewriter.finalizeRootUpdate(user);
      modified = true;
    }

    if (modified)
      return success();

    return rewriter.notifyMatchFailure(
        op, "No foldable concats found around QDQ node.");
  }
};
} // namespace

void xilinx::xten::populateXTenQDQConcatPatterns(RewritePatternSet &patterns) {
  patterns.add<XTenQDQConcat>(patterns.getContext());
}
