//===- XTenNNBase.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/IR/XTenNN.h"
#include "xten/Dialect/XTenNN/IR/XTenNNAttributes.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace amd::xten_nn;

#include "xten/Dialect/XTenNN/IR/XTenNNBase.cpp.inc"

void XTenNNDialect::initialize() {
  // Delegate to the registry methods.
  registerOps();
  registerAttributes();
}

Operation *XTenNNDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (isa<UnspecifiedAttr>(value))
    return builder.create<UnspecifiedOp>(loc, type);
  return nullptr;
}