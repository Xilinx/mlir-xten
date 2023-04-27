#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

add_xten_unittest(XtenNNIRTests
  TestXTenNNTypes.cpp

  DEPENDS
  XTenNNDialect
)

target_link_libraries(XtenNNIRTests
  PRIVATE
  XTenNNDialect)