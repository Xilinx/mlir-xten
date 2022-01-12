//===- linalg_conv_2d_lrelu_maxpool.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -linalg-to-dse -o /dev/null | FileCheck %s
//   CHECK:       Conv2DLreluMaxpoolOp:
//   CHECK-NEXT:        - node_name=""conv2d_lrelu_maxpool0""
//   CHECK-NEXT:        - in_dim=[1, 128, 128, 3]
//   CHECK-NEXT:        - filter_dim=[3, 3, 16]
//   CHECK-NEXT:        - stride_dim=[1, 1]
//   CHECK-NEXT:        - pad_dim=[1, 1, 1, 1]
//   CHECK-NEXT:        - postp_filter_dim=[2, 2, 16]
//   CHECK-NEXT:        - postp_stride_dim=[2, 2]
//   CHECK-NEXT:        - postp_pad_dim=[0, 0, 0, 0]
module attributes {torch.debug_module_name = "HelloWorld"}  {
  func @forward(%arg0: tensor<1x3x128x128xf32>) -> tensor<1x16x64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %1 = linalg.init_tensor [1, 16, 64, 64] : tensor<1x16x64x64xf32>
    %0 = linalg.pad_tensor %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x3x128x128xf32> to tensor<1x3x130x130xf32>
    %2 = linalg.conv_2d_lrelu_maxpool {dilation = dense<1> : tensor<2xi64>, layer_name = "conv2d_lrelu_maxpool0", mp_dilation = dense<1> : tensor<2xi64>, mp_kernel_size = dense<2> : tensor<2xi64>, mp_padding = dense<0> : tensor<4xi64>, mp_stride = dense<2> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>} ins(%0, %cst_1, %cst_3, %cst_0 : tensor<1x3x130x130xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, f32) outs(%1 : tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32>
    return %2 : tensor<1x16x64x64xf32>
  }
}
