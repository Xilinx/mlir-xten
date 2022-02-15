#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_library(XTenConversionPasses
ATenToXTenPass.cpp
Passes.cpp
XTenToAffine.cpp
XTenToLinalg.cpp

DEPENDS
XTenConversionIncGen
XTenDialect
ATenToXTenIncGen

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
)
