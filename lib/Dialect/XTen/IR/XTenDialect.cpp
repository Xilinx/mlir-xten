// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenPasses.h"
#include "xten/Dialect/XTen/XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;

void XTenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xten/Dialect/XTen/XTenOps.cpp.inc"
      >();
}
