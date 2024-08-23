#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.
if(MLIR_XTEN_ENABLE_TORCH)
    set(TorchSrc XTenNNToTorch.cpp)
    set(TorchLib TorchMLIRTorchDialect TorchMLIRTorchConversionPasses)
endif()

add_mlir_library(XTenConversionPasses
  Passes.cpp
  TosaToXTenNN.cpp
  XTenNNToTosa.cpp
  XTenNNToLinalg.cpp
  ${TorchSrc}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  XTenConversionIncGen
  
  LINK_COMPONENTS
  Core
  
  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTosaDialect
  MLIRVectorDialect
  XTenNNDialect
  ${TorchLib}
)

if(MLIR_XTEN_ENABLE_TORCH)
  target_compile_definitions(obj.XTenConversionPasses PRIVATE MLIR_XTEN_ENABLE_TORCH=1)
endif()
