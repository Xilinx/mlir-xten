//===- CanonicalizePass.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Declares the XTenNN canonicalize pass.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace amd::xten_nn {
#define GEN_PASS_DECL_XTENNNCANONICALIZEPASS
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"

/// Populate \p patterns with XTen concat folding patterns.
/// WARNING: This is a destructive pattern, which does not preserve the
///          semantics of the IR. This is required to enable folding that other
///          parts of our compiler depend on at this time.
void populateQDQConcatPatterns(mlir::RewritePatternSet &patterns);

/// Populate \p patterns with all common XTen canonicalization patterns.
/// These include all stand-alone patterns.
void populateCanonicalizePatterns(mlir::RewritePatternSet &patterns);

/// Populate \p patterns with destructive, non-semantic-preserving patterns.
/// WARNING: Applying these patterns will change the semantics of the IR. Only
///          use them if you are sure that you need them.
void populateDestructiveCanonicalizePatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createCanonicalizePass();
std::unique_ptr<mlir::Pass>
createCanonicalizePass(bool allowDestructivePatterns);
} // namespace amd::xten_nn
