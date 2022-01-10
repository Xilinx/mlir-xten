//===- LinalgToDsePass.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ATEN_DSE_INPUT_H
#define ATEN_DSE_INPUT_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLinalgToDsePass();

std::map<std::string, uint64_t> getATenOpStats(mlir::Operation *op);

} // namespace xten
} // namespace xilinx

#endif // ATEN_DSE_INPUT_H