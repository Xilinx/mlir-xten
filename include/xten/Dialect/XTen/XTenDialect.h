//===- XTenDialect.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_DIALECT_H
#define XTEN_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;

namespace xilinx {
namespace xten {

} // namespace xten
} // namespace xilinx

#include "xten/Dialect/XTen/XTenOpsDialect.h.inc"

#endif // XTEN_DIALECT_H
