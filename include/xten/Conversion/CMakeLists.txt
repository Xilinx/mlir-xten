#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XTenConversion)
add_public_tablegen_target(XTenConversionIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToXTen.td)
mlir_tablegen(ATenToXTen.cpp.inc -gen-rewriters)
add_public_tablegen_target(ATenToXTenIncGen)

add_mlir_doc(Passes XTenConversionPasses ./ -gen-pass-doc)