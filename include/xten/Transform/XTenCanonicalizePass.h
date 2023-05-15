#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace xilinx::xten {
#define GEN_PASS_DECL_XTENCANONICALIZEPASS
#include "xten/Transform/Passes.h.inc"

/// Populate \p patterns with XTen concat folding patterns.
/// WARNING: This is a destructive pattern, which does not preserve the
///          semantics of the IR. This is required to enable folding that other
///          parts of our compiler depend on at this time.
void populateXTenQDQConcatPatterns(mlir::RewritePatternSet &patterns);

/// Populate \p patterns with all common XTen canonicalization patterns.
/// These include all stand-alone patterns.
void populateXTenCanonicalizePatterns(mlir::RewritePatternSet &patterns);

/// Populate \p patterns with destructive, non-semantic-preserving patterns.
/// WARNING: Applying these patterns will change the semantics of the IR. Only
///          use them if you are sure that you need them.
void populateXTenDestructiveCanonicalizePatterns(
    mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createXTenCanonicalizePass();
std::unique_ptr<mlir::Pass>
createXTenCanonicalizePass(bool allowDestructivePatterns);
} // namespace xilinx::xten
