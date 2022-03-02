//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTen/XTenPasses.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "xten/Dialect/XTen/XTenPasses.h.inc"
}

namespace xilinx {
namespace xten {

void registerXTenPasses() {
  ::registerXTenDialectPasses();
}

}
}