//===- ATenToXTenPass.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ATEN_TO_XTEN_H
#define ATEN_TO_XTEN_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createATenToXTenPass();

}  // namespace xten
} // namespace xilinx

#endif // ATEN_TO_XTEN_H