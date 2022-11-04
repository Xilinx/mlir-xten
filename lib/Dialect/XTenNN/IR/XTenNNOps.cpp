//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#include "xten/Dialect/XTenNN/IR/XTenNNDialect.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

using namespace mlir;
using namespace amd::xten_nn;

#define GET_OP_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"
