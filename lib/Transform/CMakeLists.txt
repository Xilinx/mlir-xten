#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

if(MLIR_XTEN_WITH_TORCH)
  set(TORCH_SRCS
    ATenDialectOpStats.cpp
    ATenLoweringPass.cpp
    ATenOpReport.cpp 
    LowerToLibATenPass.cpp
  )
endif()

add_mlir_library(XTenTransformPasses
 ${TORCHLOWERATEN}

 PARTIAL_SOURCES_INTENDED

 Passes.cpp
 XTenMinimizeLiveTensors.cpp
 
 DEPENDS
 XTenTransformIncGen
 MLIRLinalgDialect
)
