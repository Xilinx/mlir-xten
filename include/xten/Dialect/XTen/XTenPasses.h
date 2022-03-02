//===- XTenPasses.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_PASSES_H
#define XTEN_PASSES_H

#include "mlir/Pass/Pass.h"

#include "xten/Dialect/XTen/XTenDataflow.h"
#include "xten/Dialect/XTen/XTenNamePass.h"

namespace xilinx {
namespace xten {

void registerXTenPasses();

}
}

#endif // XTEN_PASSES_H
