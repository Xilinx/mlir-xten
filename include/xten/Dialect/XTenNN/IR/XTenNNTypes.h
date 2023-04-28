//===- XTenNNTypes.h ---------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNNTYPES_H
#define XTENNNTYPES_H

#include <mlir/IR/BuiltinAttributes.h>
namespace amd::xten_nn {

/// Check that the floating point value is a power-of-two value.
///
/// Power-of-two value being 2^x where x is an integer value.
///
///\param attr the floating point attribute
///\return true when the floating point is a power-of-two
///\return false otherwise or when the attribute is not a FloatAttr
bool isPowerOfTwoFloat(mlir::Attribute attr);

} // namespace amd::xten_nn
#endif // XTENNNTYPES_H