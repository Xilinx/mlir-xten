//===- XTenNNAttributes.h -------------------------------------*- C++ //-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Declaration of the XTenNN dialect base.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNAttrDefs.h.inc"

//===----------------------------------------------------------------------===//
