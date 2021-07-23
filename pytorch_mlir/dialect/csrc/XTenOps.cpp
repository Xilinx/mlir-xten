// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#include "XTenDialect.h"
#include "XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;

#define GET_OP_CLASSES
#include "XTenOps.cpp.inc"
