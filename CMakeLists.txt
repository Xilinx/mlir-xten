#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(mlir-xten LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(MLIR_XTEN_ENABLE_CONVERSION_PASSES
       "Build XTen conversion passes and register them in aten-opt" OFF)

option(MLIR_XTEN_INSTALL_PACKAGE
       "Install headers and a relocatable CMake config package for mlir-xten" OFF)

function(detect_pybind11_install)
  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(CHECK_FAIL "not found (install via 'pip install pybind11' or set pybind11_DIR)")
      return()
    endif()
    set(pybind11_DIR "${PACKAGE_DIR}" PARENT_SCOPE)
  endif()
endfunction()
detect_pybind11_install()
find_package(pybind11 REQUIRED)
set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIR})
find_package(Python3 COMPONENTS Interpreter Development)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

else()
  message(STATUS "Building MLIR-XTEN in-tree")
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(XTEN_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(XTEN_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${XTEN_SOURCE_DIR}/include)
include_directories(${XTEN_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

# Create a general check-all command to run both lit and unittests
# Collects all lit tests between the begin and end function
umbrella_lit_testsuite_begin(check-xten-all)

# Make sure we build the docs
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_custom_target(docs ALL)
  add_dependencies(docs mlir-doc)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
add_subdirectory(test)

umbrella_lit_testsuite_end(check-xten-all)

# ##############################################################################
# Installed CMake config package
#
# Downstream projects have had to reach into this project's build tree via
# XTEN_DIR / MLIR_XTEN_BINARY_DIR and a hand-written Findmlir-xten.cmake.
# Exporting a relocatable config package lets them use
# find_package(mlir-xten CONFIG) instead.
#
# Only the namespaced xten:: targets are published.  Do not add a Find module
# alongside this: a plain, non-namespaced target of the same name leaks into
# INTERFACE_LINK_LIBRARIES and degrades to a bare -lXTenNNDialect link flag.
#
# XTenCAPI is deliberately not exported.  It is built with
# add_mlir_public_c_api_library(... ENABLE_AGGREGATION), whose object/aggregate
# machinery does not survive install(EXPORT), and no downstream consumer links
# it -- it exists for the Python bindings, which are built in-tree.
# ##############################################################################
if(MLIR_XTEN_INSTALL_PACKAGE)
  include(CMakePackageConfigHelpers)

  set(XTEN_INSTALL_INCLUDEDIR include)
  set(XTEN_INSTALL_PACKAGEDIR lib/cmake/mlir-xten)

  set(XTEN_EXPORT_TARGETS XTenNNInterfaces XTenNNDialect XTenNNTransforms)
  if(MLIR_XTEN_ENABLE_CONVERSION_PASSES)
    list(APPEND XTEN_EXPORT_TARGETS XTenConversionPasses)
  endif()

  # These targets get their include paths from the directory-scoped
  # include_directories() above, so they carry no INTERFACE_INCLUDE_DIRECTORIES
  # of their own.  Add the install-tree one only -- adding a BUILD_INTERFACE
  # entry is unnecessary in-tree and an absolute source path would make
  # install(EXPORT) fail.
  foreach(_xten_target IN LISTS XTEN_EXPORT_TARGETS)
    target_include_directories(
      ${_xten_target} INTERFACE
      $<INSTALL_INTERFACE:${XTEN_INSTALL_INCLUDEDIR}>)
  endforeach()

  install(
    TARGETS ${XTEN_EXPORT_TARGETS}
    EXPORT mlir-xtenTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION ${XTEN_INSTALL_INCLUDEDIR})

  # Checked-in headers.
  install(
    DIRECTORY ${XTEN_SOURCE_DIR}/include/xten ${XTEN_SOURCE_DIR}/include/xten-c
    DESTINATION ${XTEN_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.td")

  # TableGen'd headers, which mirror the source layout under the build tree.
  install(
    DIRECTORY ${XTEN_BINARY_DIR}/include/xten
    DESTINATION ${XTEN_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inc")

  install(
    EXPORT mlir-xtenTargets
    NAMESPACE xten::
    DESTINATION ${XTEN_INSTALL_PACKAGEDIR})

  configure_package_config_file(
    ${XTEN_SOURCE_DIR}/cmake/mlir-xtenConfig.cmake.in
    ${XTEN_BINARY_DIR}/cmake/mlir-xtenConfig.cmake
    INSTALL_DESTINATION ${XTEN_INSTALL_PACKAGEDIR}
    PATH_VARS XTEN_INSTALL_INCLUDEDIR)

  install(FILES ${XTEN_BINARY_DIR}/cmake/mlir-xtenConfig.cmake
          DESTINATION ${XTEN_INSTALL_PACKAGEDIR})
endif()
