#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(aten-opt aten-opt.cpp)
llvm_update_compile_flags(aten-opt)

set(LIBS
  XTenDialect
  XTenConversionPasses
  XTenTransformPasses
  XTenTransforms
  XTenUtil
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchUtils
  MLIR
  )

target_link_libraries(aten-opt PRIVATE ${LIBS})
