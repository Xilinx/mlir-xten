//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNAttributes.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"

using namespace mlir;

#define GET_ATTRDEF_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNAttrDefs.cpp.inc"

void amd::xten_nn::XTenNNDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xten/Dialect/XTenNN/IR/XTenNNAttrDefs.cpp.inc"
      >();
}
