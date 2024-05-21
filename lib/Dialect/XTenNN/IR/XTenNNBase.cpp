//===- XTenNNBase.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/IR/XTenNN.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace amd::xten_nn;

#include "xten/Dialect/XTenNN/IR/XTenNNBase.cpp.inc"

void XTenNNDialect::initialize() {
  // Delegate to the registry methods.
  registerOps();
}