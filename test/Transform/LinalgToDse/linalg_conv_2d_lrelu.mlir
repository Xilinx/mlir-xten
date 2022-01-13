//===- linalg_conv_2d_lrelu.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -linalg-to-dse -o /dev/null | FileCheck %s
//   CHECK:       "filter_dim": [
//   CHECK:         3,
//   CHECK:         3,
//   CHECK:         16
//   CHECK:       ],
//   CHECK:       "in_dim": [
//   CHECK:         1,
//   CHECK:         64,
//   CHECK:         64,
//   CHECK:         16
//   CHECK:       ],
//   CHECK:       "node_name": "\"conv2d_lrelu0\"",
//   CHECK:       "pad_dim": [
//   CHECK:         1,
//   CHECK:         1,
//   CHECK:         1,
//   CHECK:         1
//   CHECK:       ],
//   CHECK:       "stride_dim": [
//   CHECK:         1,
//   CHECK:         1
//   CHECK:       ],
module attributes {torch.debug_module_name = "HelloWorld"}  {
  func @forward(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<16x16x3x3xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %1 = linalg.init_tensor [1, 16, 64, 64] : tensor<1x16x64x64xf32>
    %3 = linalg.pad_tensor %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x16x64x64xf32> to tensor<1x16x66x66xf32>
    %4 = linalg.conv_2d_lrelu {dilation = dense<1> : tensor<2xi64>, layer_name = "conv2d_lrelu0", stride = dense<1> : tensor<2xi64>} ins(%3, %cst_2, %cst_3, %cst_0 : tensor<1x16x66x66xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, f32) outs(%1 : tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32>
    return %4 : tensor<1x16x64x64xf32>
  }
}
