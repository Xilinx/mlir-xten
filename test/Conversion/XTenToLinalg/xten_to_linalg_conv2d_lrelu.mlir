//===- xten_to_linalg_conv2d_lrelu.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK:linalg.conv_2d_lrelu {dilation = dense<1> : tensor<2xi64>, layer_name = "conv2d_lrelu0", stride = dense<1> : tensor<2xi64>} ins({{.*}}, {{.*}}, {{.*}} : tensor<1x3x130x130xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, f32) outs({{.*}} : tensor<1x16x128x128xf32>)
module attributes {torch.debug_module_name = "HelloWorld"}  {
  func @forward(%arg0: !torch.vtensor<[1,3,128,128],f32>) -> !torch.vtensor<[1,16,128,128],f32> {
    %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<16xf32>) : !torch.vtensor<[16],f32>
    %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<16x3x3x3xf32>) : !torch.vtensor<[16,3,3,3],f32>
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = "xten.conv2d_lrelu"(%arg0, %1, %0, %2, %2, %2, %int1, %float1.000000e-01) {layer_name = "conv2d_lrelu0"} : (!torch.vtensor<[1,3,128,128],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float) -> !torch.vtensor<[1,16,128,128],f32>
    return %3 : !torch.vtensor<[1,16,128,128],f32>
  }
}