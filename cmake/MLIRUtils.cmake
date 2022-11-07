#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022 Advanced Micro Devices, Inc.

function(mlir_gen_iface prefix iface kind)
    set(LLVM_TARGET_DEFINITIONS ${iface}.td)

    mlir_tablegen(${iface}.h.inc -gen-${kind}-interface-decls)
    mlir_tablegen(${iface}.cpp.inc -gen-${kind}-interface-defs)

    add_public_tablegen_target(${prefix}${iface}InterfaceIncGen)
    add_dependencies(${prefix}IncGen ${prefix}${iface}InterfaceIncGen)

    add_mlir_doc(${iface} ${iface} Interfaces/ -gen-${kind}-interface-docs)
endfunction()

function(mlir_gen_ir prefix)
    string(TOLOWER ${prefix} filter)

    set(LLVM_TARGET_DEFINITIONS ${prefix}Ops.td)

    mlir_tablegen(${prefix}Base.h.inc -gen-dialect-decls)
    mlir_tablegen(${prefix}Base.cpp.inc -gen-dialect-defs)
    mlir_tablegen(${prefix}Ops.h.inc -gen-op-decls)
    mlir_tablegen(${prefix}Ops.cpp.inc -gen-op-defs)

    add_public_tablegen_target(${prefix}IRIncGen)
    add_dependencies(${prefix}IncGen ${prefix}IRIncGen)

    add_mlir_doc(${prefix}Ops ${prefix} Dialects/ -gen-dialect-doc)
endfunction()