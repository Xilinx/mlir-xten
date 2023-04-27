#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2023 Advanced Micro Devices, Inc.

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  aten-opt
  )

add_lit_testsuite(check-aten "Running the aten regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-aten PROPERTIES FOLDER "Tests")

add_lit_testsuites(ATEN ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-all check-aten)

# Add a target to run the unittests for XTen
add_custom_target(XTenUnitTests)
set_target_properties(XTenUnitTests PROPERTIES FOLDER "XTen Unittests")
function(add_xten_unittest test_dirname)
  add_unittest(XTenUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(lib)