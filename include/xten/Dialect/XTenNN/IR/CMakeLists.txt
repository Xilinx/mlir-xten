#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

set(LLVM_TARGET_DEFINITIONS XTenNNOps.td)

mlir_tablegen(XTenNNBase.h.inc -gen-dialect-decls)
mlir_tablegen(XTenNNBase.cpp.inc -gen-dialect-defs)
mlir_tablegen(XTenNNOps.h.inc -gen-op-decls)
mlir_tablegen(XTenNNOps.cpp.inc -gen-op-defs)
mlir_tablegen(XTenNNTypes.h.inc -gen-typedef-decls -typedefs-dialect=xten_nn)
mlir_tablegen(XTenNNTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=xten_nn)

add_public_tablegen_target(XTenNNDialectIncGen)
add_dependencies(XTenNNIncGen XTenNNDialectIncGen)

add_mlir_doc(XTenNNOps XTenNN Dialects/ -gen-dialect-doc)
