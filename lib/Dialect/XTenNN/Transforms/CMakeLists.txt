#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

add_mlir_dialect_library(XTenNNTransforms
  Simplify.cpp
  CanonicalizePass.cpp
  QDQConcat.cpp

  DEPENDS
  XTenNNIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  XTenNNDialect
)