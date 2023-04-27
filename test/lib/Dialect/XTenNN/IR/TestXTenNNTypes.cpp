//===-- TestXTenNNTypes.cpp - Test XTenNN Type Definitions*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNTypes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <gtest/gtest.h>

namespace {

struct TestValue {
  float value;
  bool expectedResult;
};

std::ostream &operator<<(std::ostream &os, const TestValue &value) {
  os << "{ value = " << value.value << ", expected = " << value.expectedResult
     << " }";
  return os;
}

class PowerOfTwoTest : public testing::TestWithParam<TestValue> {};

auto getPowerOfTwoTestCases() {
  mlir::MLIRContext context;
  return testing::Values<TestValue>(
      TestValue({1.0, true}), TestValue({0.125, true}),
      TestValue({0.001, false}), TestValue({16.0, true}),
      TestValue({-0.5, false}),
      TestValue({std::numeric_limits<float>::infinity(), false}),
      TestValue({-std::numeric_limits<float>::infinity(), false}),
      // Epsilon described as 2^-23 (difference between 1.0 and the next
      // representable value)
      TestValue({std::numeric_limits<float>::epsilon(), true}),
      TestValue({std::numeric_limits<float>::lowest(), false}),
      // Min is described as 2^-126
      TestValue({std::numeric_limits<float>::min(), true}),
      TestValue({std::numeric_limits<float>::max(), false}),
      // Denorm Min is described as 2^-149
      TestValue({std::numeric_limits<float>::denorm_min(), true}),
      TestValue({std::numeric_limits<float>::quiet_NaN(), false}),
      TestValue({std::numeric_limits<float>::signaling_NaN(), false}));
}

TEST_P(PowerOfTwoTest, CheckValues) {
  TestValue testCase = GetParam();

  // Attributes need a live context to be able to generate.
  // Here we generate the attribute given the test parameter value.
  mlir::MLIRContext context;
  auto attr =
      mlir::FloatAttr::get(mlir::Float32Type::get(&context), testCase.value);

  EXPECT_EQ(testCase.expectedResult, amd::xten_nn::isPowerOfTwoFloat(attr));
}

TEST(PowerOfTwoNonFloatTest, NonFloatAttr) {
  mlir::MLIRContext context;
  auto attr = mlir::StringAttr::get(&context, "testvalue");

  // We should always return false when the attribute is not the expected type
  // of FloatAttr
  EXPECT_FALSE(amd::xten_nn::isPowerOfTwoFloat(attr));
}

INSTANTIATE_TEST_SUITE_P(FloatValues, PowerOfTwoTest, getPowerOfTwoTestCases());

} // namespace