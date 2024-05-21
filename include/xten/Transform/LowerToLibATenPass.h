//===- LowerToLibATenPass.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2020 - 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef LOWER_TO_LIBATEN_PASS_H
#define LOWER_TO_LIBATEN_PASS_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerToLibATenPass();

} // namespace xten
} // namespace xilinx

#endif // LOWER_TO_LIBATEN_PASS_H