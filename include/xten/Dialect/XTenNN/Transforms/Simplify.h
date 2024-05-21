//===- Simplify.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Declares the Simplify pass.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include "xten/Dialect/XTenNN/IR/XTenNN.h"

namespace mlir {
class RewritePatternSet;
} // namespace mlir

namespace amd::xten_nn {

#define GEN_PASS_DECL_XTENNNSIMPLIFY
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"

/// Obtains the graph simplification patterns.
void populateSimplifyPatterns(mlir::RewritePatternSet &patterns);

/// Creates the graph simplification pass.
std::unique_ptr<mlir::Pass> createSimplifyPass();
} // namespace amd::xten_nn
