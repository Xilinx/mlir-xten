// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "aten/Dialect/XTen/XTenDialect.h"
#include "aten/Dialect/XTen/XTenPasses.h"
#include "aten/Dialect/XTen/XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;

void XTenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aten/Dialect/XTen/XTenOps.cpp.inc"
      >();
}
