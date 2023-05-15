#include "xten/Transform/XTenCanonicalizePass.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "xten-canonicalize"

namespace xilinx::xten {
using namespace mlir;
#define GEN_PASS_DEF_XTENCANONICALIZEPASS
#include "xten/Transform/Passes.h.inc"
} // namespace xilinx::xten

using namespace mlir;
using namespace xilinx;
using namespace xilinx::xten;

namespace {
struct XTenCanonicalizePass
    : public xten::impl::XTenCanonicalizePassBase<XTenCanonicalizePass> {
  using XTenCanonicalizePassBase::XTenCanonicalizePassBase;

  XTenCanonicalizePass(bool allowDestructivePatterns) {
    this->allowDestructivePatterns = allowDestructivePatterns;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    populateXTenCanonicalizePatterns(patterns);
    if (allowDestructivePatterns)
      populateXTenDestructiveCanonicalizePatterns(patterns);

    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed())
      signalPassFailure();
  }
};
} // namespace

void xilinx::xten::populateXTenCanonicalizePatterns(
    RewritePatternSet & /*patterns*/) {
  // TODO: FXML-2034 add non-destructive patterns
}

void xilinx::xten::populateXTenDestructiveCanonicalizePatterns(
    RewritePatternSet &patterns) {
  // Add destructive canonicalization patterns here
  populateXTenQDQConcatPatterns(patterns);
}

std::unique_ptr<Pass> xilinx::xten::createXTenCanonicalizePass() {
  return std::make_unique<XTenCanonicalizePass>();
}

std::unique_ptr<Pass>
xilinx::xten::createXTenCanonicalizePass(bool allowDestructivePatterns) {
  return std::make_unique<XTenCanonicalizePass>(allowDestructivePatterns);
}
