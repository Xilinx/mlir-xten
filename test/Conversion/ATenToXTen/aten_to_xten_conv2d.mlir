//===- aten_to_xten_conv2d.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --aten-to-xten | FileCheck %s
// CHECK: "xten.conv2d"(%arg0, %arg1, %arg2
module  {
  func @graph(%arg0: tensor<1x2x128x128xf32>, %arg1: tensor<16x2x7x7xf32>, %arg2: tensor<16xf32>) -> tensor<1x16x64x64xf32> {
    %c2_i64 = constant 2 : i64
    %c3_i64 = constant 3 : i64
    %false = constant false
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %0 = basicpy.build_list %c2_i64, %c2_i64 : (i64, i64) -> !basicpy.ListType
    %1 = basicpy.build_list %c3_i64, %c3_i64 : (i64, i64) -> !basicpy.ListType
    %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
    %3 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
    %4 = "aten.convolution"(%arg0, %arg1, %arg2, %0, %1, %2, %false, %3, %c1_i64) : (tensor<1x2x128x128xf32>, tensor<16x2x7x7xf32>, tensor<16xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64) -> tensor<1x16x64x64xf32>
    return %4 : tensor<1x16x64x64xf32>
  }
}