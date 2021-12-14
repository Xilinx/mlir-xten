//===- air_to_linalg_conv2d_relu.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK:linalg.conv_2d_relu {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins({{.*}}, {{.*}} : tensor<1x27x49x42xf32>, tensor<28x27x3x3xf32>) outs({{.*}} : tensor<1x28x45x40xf32>)
module attributes {torch.debug_module_name = "model"}  {
  func @forward(%arg0: !torch.vtensor<[1,27,49,42],f32>, %arg1: !torch.vtensor<[28,27,3,3],f32>, %arg2: !torch.vtensor<[28],f32>) -> !torch.vtensor<[1,28,45,40],f32> {
    %int2 = torch.constant.int 2 
    %int1 = torch.constant.int 1 
    %int0 = torch.constant.int 0 
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %3 = torch.prim.ListConstruct %int0, %int0,%int0, %int0 : (!torch.int, !torch.int,!torch.int, !torch.int) -> !torch.list<!torch.int>
    %4 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %5 = "xten.conv2d_relu"(%arg0, %arg1, %arg2, %2, %3, %4, %int1) : (!torch.vtensor<[1,27,49,42],f32>, !torch.vtensor<[28,27,3,3],f32>, !torch.vtensor<[28],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.vtensor<[1,28,45,40],f32>
    return %5 : !torch.vtensor<[1,28,45,40],f32>
  }
}
