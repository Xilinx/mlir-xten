#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

cmake_minimum_required(VERSION 3.10)

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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

function(TORCH_MLIR_detect_pybind11_install)
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
TORCH_MLIR_detect_pybind11_install()
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

  link_directories(${TORCH_MLIR_LIB_DIR})
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

# TORCH_MLIR
include_directories(${TORCH_MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

# Create a general check-all command to run both lit and unittests
# Collects all lit tests between the begin and end function
umbrella_lit_testsuite_begin(check-xten-all)

# Make sure we build the docs
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
add_custom_target(docs ALL)
add_dependencies(docs mlir-doc)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
add_subdirectory(test)

umbrella_lit_testsuite_end(check-xten-all)
