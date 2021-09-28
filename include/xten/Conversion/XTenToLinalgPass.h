//===- XTenToLinalgPass.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_TO_LINALG_H
#define XTEN_TO_LINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createXTenToLinalgPass();

} // namespace xten
} // namespace xilinx

#endif // XTEN_TO_LINALG_H