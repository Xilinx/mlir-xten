//===- XTenPasses.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENPASSES_H_
#define XTENPASSES_H_

#include "mlir/Pass/Pass.h"

#include "xten/Dialect/XTen/XTenDataflow.h"
//#include "xten/Dialect/XTen/XTenToAffinePass.h"
#include "xten/Dialect/XTen/XTenNamePass.h"

namespace xilinx {
  namespace xten {
// #define GEN_PASS_CLASSES
// #include "XTenPasses.h.inc"

    void registerXTenPasses();
  }
}
#endif // XTENPASSES_H_
