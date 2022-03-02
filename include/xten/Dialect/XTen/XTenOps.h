//===- XTenOps.h ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENOPS_H
#define XTENOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#define GET_OP_CLASSES
#include "xten/Dialect/XTen/XTenOps.h.inc"

#endif // XTENOPS_H
