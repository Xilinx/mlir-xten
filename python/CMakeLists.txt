#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
)
add_library(_xten SHARED
  XTenModule.cpp
)
set_target_properties(_xten PROPERTIES PREFIX "")
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(_xten
  PRIVATE
  XTenTransformPasses
  XTenConversionPasses
  XTenTransforms
  XTenUtil
  XTenDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchUtils
  ${Python3_LIBRARIES}
  ${dialect_libs}
  ${conversion_libs}
  MLIRPass
  MLIRIR
)

install(TARGETS _xten
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

install(DIRECTORY xten
DESTINATION python COMPONENT python)

#add_subdirectory(test)
