//===- air_to_linalg_conv2d.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK: linalg.conv_2d_nchw {__internal_linalg_transform__ = "xten_conv2d"} ins({{.*}}, {{.*}} : memref<16x16x64x64xf32>, memref<16x16x1x1xf32>) outs({{.*}} : memref<16x16x64x64xf32>)
module  {
  func @conv2d(%arg0: tensor<16x16x1x1xf32>, %arg1: tensor<16x16x64x64xf32>) -> tensor<16x16x64x64xf32> {
    %false = constant false
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %0 = basicpy.singleton : !basicpy.NoneType
    %1 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
    %2 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
    %3 = "xten.conv2d"(%arg1, %arg0, %0, %1, %2, %1, %false, %2, %c1_i64) : (tensor<16x16x64x64xf32>, tensor<16x16x1x1xf32>, !basicpy.NoneType, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64) -> tensor<16x16x64x64xf32>
    return %3 : tensor<16x16x64x64xf32>
  }
}