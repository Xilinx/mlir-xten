//===- XTenNNDialect.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNDialect.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "xten/Dialect/XTenNN/IR/XTenNNOpsDialect.cpp.inc"

using namespace mlir;
using namespace amd::xten_nn;

void XTenNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"
      >();
}
