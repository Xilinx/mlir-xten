//===-- XTenNNTypes.cpp - XTenNN Type definitions *------- tablegen -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNTypes.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/bit.h>

using namespace mlir;
using namespace llvm;

namespace amd::xten_nn {

bool isPowerOfTwoFloat(Attribute attr) {
  // Only handle floating point attributes
  if (!attr.isa<FloatAttr>()) {
    return false;
  }
  auto floatValue = attr.cast<FloatAttr>().getValue();

  // Negative values, non finite and not IEEE cannot be handled.
  if (!floatValue.isFinite() || !floatValue.isIEEE() ||
      floatValue.isNegative()) {
    return false;
  }

  // Here we convert the float value to its components with the following
  // equation:
  //     floatValue = normalized_mantissa * 2^exponent;
  int exponent = 0;
  APFloat mantissa =
      frexp(floatValue, exponent, APFloat::roundingMode::TowardZero);

  // Check that the mantissa is zero. If the normalized mantissa value is 0.5
  // then the mantissa was zero.
  return mantissa.convertToFloat() == 0.5;
}

} // namespace amd::xten_nn