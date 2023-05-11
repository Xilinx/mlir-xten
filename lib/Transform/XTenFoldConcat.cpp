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
struct XTenFoldConcat : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t axis = op.getAxis();

    // List of operands for the new concat. There will be at least the same
    // number of operands as the original op
    SmallVector<Value, 8> concatOperands;
    concatOperands.reserve(op->getNumOperands());

    bool canFold = false;
    for (Value operand : op->getOperands()) {
      // Keep track of original operands, so that we can create a new one later
      concatOperands.emplace_back(operand);

      // Foldable concats might have QDQ nodes in between
      Value qdqInput;
      if (matchPattern(operand, m_Op<DequantizeOp>(m_Op<QuantizeOp>(
                                    matchers::m_Any(&qdqInput))))) {
        operand = qdqInput;
      }

      auto producer = dyn_cast_or_null<ConcatOp>(operand.getDefiningOp());
      if (producer == nullptr)
        continue;

      // We can only fold the concat if the axis is the same
      uint64_t producerAxis = producer.getAxis();
      if (axis != producerAxis)
        continue;

      // Foldable concat found. Take all incoming operands and use it. Make sure
      // to replace the original operand. No need to reserve space, as append
      // does that for us
      canFold = true;
      Operation::operand_range producerOperands = producer->getOperands();
      concatOperands.pop_back();
      concatOperands.append(producerOperands.begin(), producerOperands.end());
    }

    // No foldable concats found
    if (!canFold)
      return rewriter.notifyMatchFailure(op, "No foldable concats found");

    // Replace the original concat with a new one that contains the original and
    // folded operands.
    rewriter.replaceOpWithNewOp<ConcatOp>(op, op->getResultTypes(),
                                          concatOperands, axis);
    return success();
  }
};
} // namespace

void xilinx::xten::populateXTenFoldConcatPatterns(RewritePatternSet &patterns,
                                                  PatternBenefit benefit) {
  patterns.add<XTenFoldConcat>(patterns.getContext(), benefit);
}
