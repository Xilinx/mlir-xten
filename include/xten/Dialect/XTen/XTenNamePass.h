//===- XTenNamePass.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_NAME_PASS_H
#define XTEN_NAME_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createXTenNamePass();

} // namespace xten
} // namespace xilinx

#endif
