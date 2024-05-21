#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

set(LLVM_TARGET_DEFINITIONS EnclaveOpInterface.td)

mlir_tablegen(EnclaveOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(EnclaveOpInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(XTenNNEnclaveOpInterfaceInterfaceIncGen)
add_dependencies(XTenNNIncGen XTenNNEnclaveOpInterfaceInterfaceIncGen)

add_mlir_doc(EnclaveOpInterface EnclaveOpInterface Interfaces/ -gen-op-interface-docs)
