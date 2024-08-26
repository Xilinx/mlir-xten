//===- XTenNNOps.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTENNNOPS_H
#define XTENNNOPS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterfaces.h"

namespace mlir::OpTrait {
template <typename ConcreteType>
class TosaExtension : public TraitBase<ConcreteType, TosaExtension> {};

template <typename ConcreteType>
class ElementwiseUnary : public TraitBase<ConcreteType, ElementwiseUnary> {};

template <typename ConcreteType>
class ElementwiseBinary : public TraitBase<ConcreteType, ElementwiseBinary> {};
} // namespace mlir::OpTrait

#define GET_OP_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h.inc"

#endif // XTENNNOPS_H
