//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRRtDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "AIRRtOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx::airrt;

void AIRRtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AIRRtOps.cpp.inc"
      >();
  addTypes<TensorType>();
}
