#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2023 Advanced Micro Devices, Inc.

add_mlir_library(XTenConversionPasses
ATenToXTenPass.cpp
Passes.cpp
TosaToXTenNN.cpp
XTenToAffine.cpp
XTenToLinalg.cpp
XTenNNToTosa.cpp

DEPENDS
XTenConversionIncGen
XTenDialect

# For more information on why these two libraries are commented out refer to:
# - include/xten/Conversion/ATenToXTen.td
# - include/xten/Conversion/XTenFusions.td

# ATenToXTenIncGen 
# ATenToXTenIncGen2

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
)
