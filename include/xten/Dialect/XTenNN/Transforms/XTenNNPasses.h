//===- XTenNNPasses.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the XTenNN pass entry points.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include "xten/Dialect/XTenNN/IR/XTenNN.h"

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace amd::xten_nn {

/// Obtains the graph simplification patterns.
void populateSimplifyPatterns(mlir::RewritePatternSet &patterns);

/// Creates the graph simplification pass.
std::unique_ptr<mlir::Pass> createSimplifyPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "xten/Dialect/XTenNN/Transforms/XTenNNPasses.h.inc"

} // namespace amd::xten_nn