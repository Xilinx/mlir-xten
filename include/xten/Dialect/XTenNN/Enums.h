//===- Enums.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the DLNN dialect enums.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace amd::xten_nn {

/// Enumeration of known scalar operations.
///
/// This is a non-exhaustive list of supported scalar operations that we expect
/// to appear in XTenNN graphs. This enumeration exists primarily to:
///     (1) Classify operations on scalars in generic operator nodes.
///     (2) Define the ScalarInterface, which materializes required ops.
///
/// This list was initially populated with the ONNX scalar operators:
///     https://github.com/onnx/onnx/blob/main/docs/Operators.md
enum class ScalarOpKind {
  /// Unknown scalar operation.
  Unknown = 0,

  Abs,
  Acos,
  Acosh,
  Add,
  And,
  Asin,
  Asinh,
  Atan,
  Atanh,
  Cos,
  Cosh,
  Ceil,
  Div,
  Erf,
  Exp,
  Floor,
  Log,
  Max,
  Min,
  Mod,
  Mul,
  Neg,
  Not,
  Or,
  Pow,
  Reciprocal,
  Round,
  Sin,
  Sinh,
  Sub,
  Tan,
  Tanh,
  Xor
};

} // namespace amd::xten_nn

//===- Generated includes -------------------------------------------------===//

#include "xten/Dialect/XTenNN/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace amd::xten_nn {

// raw_ostream& operator<<(...)

} // namespace amd::xten_nn

namespace mlir {

// template<class> FieldParser<...>

} // namespace mlir