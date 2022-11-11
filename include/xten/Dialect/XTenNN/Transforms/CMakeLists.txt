#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022 Advanced Micro Devices, Inc.

set(LLVM_TARGET_DEFINITIONS XTenNNPasses.td)

mlir_tablegen(XTenNNPasses.h.inc -gen-pass-decls -name XTenNN)
add_public_tablegen_target(XTenNNPassesIncGen)
add_dependencies(XTenNNIncGen XTenNNPassesIncGen)

add_mlir_doc(XTenNNPasses XTenNNPasses ./ -gen-pass-doc)