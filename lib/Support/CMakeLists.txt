#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

if(${TORCH_FOUND})
  include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
  )

  set(ATEN_LIBS ${TORCH_LIBRARIES})
  add_compile_definitions(ATEN_OPS_ENABLE_TORCH)
endif()

add_definitions("-Wno-string-conversion")
add_definitions("-Wno-covered-switch-default")

add_library(aten_ops SHARED
  aten_ops.cpp
)

install(TARGETS aten_ops
LIBRARY DESTINATION lib COMPONENT python
ARCHIVE DESTINATION lib COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

target_link_libraries(aten_ops
  ${ATEN_LIBS}
  pthread
 )
