#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

add_mlir_library(XTenConversionPasses
Passes.cpp
TosaToXTenNN.cpp
XTenNNToTosa.cpp
XTenNNToLinalg.cpp
XTenNNToTorch.cpp

DEPENDS
XTenConversionIncGen

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
)
