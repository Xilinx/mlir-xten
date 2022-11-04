//===- XTenNNDialect.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_DIALECT_H
#define XTENNN_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace amd {
namespace xten_nn {} // namespace xten_nn
} // namespace amd

#include "xten/Dialect/XTenNN/IR/XTenNNOpsDialect.h.inc"

#endif // XTENNN_DIALECT_H
