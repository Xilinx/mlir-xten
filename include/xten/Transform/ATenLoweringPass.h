//===- ATenLoweringPass.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ATEN_LOWERING_PASS_H
#define ATEN_LOWERING_PASS_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createATenLoweringPass();

} // namespace xten
} // namespace xilinx

#endif // ATEN_LOWERING_PASS_H
