//===- XTenNNTypes.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
namespace amd::xten_nn {

struct HasAxisInterfaceTrait {
  struct Concept {
    virtual ~Concept();
    virtual uint64_t getAxis(Type *type) const = 0;
  };

  template <typename ConcreteType>
  struct Model : public Concept {
    unsigned getAxis(Type *type) const final {
      return llvm::cast<ConcreteType>(type)->getAxis();
    }
  };
};

template <typename ConcreteType>
class HasAxisInterface
    : public mlir::TypeInterface<ConcreteType, HasAxisInterfaceTrait> {
public:
  using mlir::TypeInterface<ConcreteType, HasAxisInterfaceTrait>::TypeInterface;
  uint64_t getAxis() const {
    return getImpl()->getAxis(this);
  }
};
} // namespace amd::xten_nn
#define GET_TYPEDEF_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNTypes.h.inc"
