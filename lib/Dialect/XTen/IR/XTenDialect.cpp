//===- XTenDialect.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenPasses.h"
#include "xten/Dialect/XTen/XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;

void XTenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xten/Dialect/XTen/XTenOps.cpp.inc"
      >();
}
