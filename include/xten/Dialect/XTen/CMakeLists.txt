#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(dialect XTenOps)
set(dialect_namespace xten)
set(LLVM_TARGET_DEFINITIONS XTenOps.td)

mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XTenOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XTenDialect)
add_public_tablegen_target(XTenDialectPassesIncGen)

#add_mlir_doc(Passes XTenDialectPasses ./ -gen-pass-doc)
