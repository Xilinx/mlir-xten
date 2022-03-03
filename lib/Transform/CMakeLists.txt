#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_library(XTenTransformPasses
  ATenDialectOpStats.cpp
  ATenLoweringPass.cpp
  ATenOpReport.cpp
  ATenVisualGraph.cpp
  LowerToLibATenPass.cpp
  Passes.cpp

  DEPENDS
  XTenTransformIncGen

  LINK_LIBS
  XTenDialect
  )
