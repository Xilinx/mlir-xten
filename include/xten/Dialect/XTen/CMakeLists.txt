#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.


add_mlir_dialect(XTenOps xten)
add_mlir_doc(XTenOps XTenDialect ./ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS XTenOps.td)
mlir_tablegen(XTenEnums.h.inc -gen-enum-decls)
mlir_tablegen(XTenEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XTenEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS XTenPasses.td)
mlir_tablegen(XTenPasses.h.inc -gen-pass-decls -name XTenDialect)
add_public_tablegen_target(XTenPassesIncGen)

add_mlir_doc(XTenPasses XTenDialectPasses ./ -gen-pass-doc)
