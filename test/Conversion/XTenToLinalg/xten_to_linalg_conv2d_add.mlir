//===- xten_to_linalg_conv2d_add.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK: linalg.conv_2d_tensor_add {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins({{.*}}, {{.*}}, {{.*}}, {{.*}} : tensor<1x2x130x130xf32>, tensor<16x2x3x3xf32>, tensor<16xf32>, tensor<1x16x128x128xf32>) outs({{.*}} : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>

module attributes {torch.debug_module_name = "model"} {
  func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,16,128,128],f32> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.vtensor.literal(dense<0.0> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>

    %bias = torch.vtensor.literal(dense<1.0> : tensor<16xf32>) : !torch.vtensor<[16],f32>

    %1 = torch.vtensor.literal(dense<0.0> : tensor<1x2x256x256xf32>) : !torch.vtensor<[1,2,256,256],f32>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = "xten.conv2d"(%1, %0, %bias, %2, %2, %3, %int1) : (!torch.vtensor<[1,2,256,256],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,16,128,128],f32>
    %5 = "xten.conv2d_tensoradd"(%arg0, %0, %bias, %2, %2, %2, %int1, %4) : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,16,128,128],f32>) -> !torch.vtensor<[1,16,128,128],f32>
    
    return %5 : !torch.vtensor<[1,16,128,128],f32>
  }
}