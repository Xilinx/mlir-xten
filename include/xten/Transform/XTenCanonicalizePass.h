#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace xilinx::xten {
#define GEN_PASS_DECL_XTENCANONICALIZEPASS
#include "xten/Transform/Passes.h.inc"

/// Populate \p patterns with XTen concat folding patterns.
void populateXTenFoldConcatPatterns(mlir::MLIRContext *ctx,
                                    mlir::RewritePatternSet &patterns);

/// Populate \p patterns with all common XTen canonicalization patterns.
/// These include all stand-alone patterns.
void populateXTenCanonicalizePatterns(mlir::MLIRContext *ctx,
                                      mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createXTenCanonicalizePass();
} // namespace xilinx::xten
