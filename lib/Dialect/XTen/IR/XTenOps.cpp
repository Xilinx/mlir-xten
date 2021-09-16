// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#include "aten/Dialect/XTen/XTenDialect.h"
#include "aten/Dialect/XTen/XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;

#define GET_OP_CLASSES
#include "aten/Dialect/XTen/XTenOps.cpp.inc"
