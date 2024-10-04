#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(AddMLIRPython)
  declare_mlir_python_sources(XTenPythonSources)
  declare_mlir_python_sources(XTenPythonSources.Dialects
    ADD_TO_PARENT XTenPythonSources)

  declare_mlir_dialect_python_bindings(
      ADD_TO_PARENT XTenPythonSources.Dialects
      ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
      TD_FILE dialects/XTenNNOps.td
      SOURCES
        dialects/xten_nn.py
      DIALECT_NAME xten_nn)
endif()
