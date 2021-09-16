// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef XTENOPS_H
#define XTENOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"

#define GET_OP_CLASSES
#include "aten/Dialect/XTen/XTenOps.h.inc"

#endif // XTENOPS_H
