//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTEN_TRANSFORM_PASSDETAIL_H
#define XTEN_TRANSFORM_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

using namespace mlir;

#define GEN_PASS_CLASSES
#include "xten/Transform/Passes.h.inc"

} // namespace xten
} // namespace xilinx

#endif // XTEN_TRANSFORM_PASSDETAIL_H