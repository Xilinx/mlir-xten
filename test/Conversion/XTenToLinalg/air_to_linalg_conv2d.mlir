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
// CHECK: %[[M_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
// CHECK: %[[CST:.+]] = arith.constant 0
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[B_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<1x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<1x45x40x28xf32>)
// CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<1x45x40x28xf32>) outs(%[[B_IN]] : tensor<1x45x40x28xf32>)
// CHECK:   arith.addf
// CHECK:   linalg.yield %12 : f32
module attributes {torch.debug_module_name = "model"}  {
  func @forward(%arg0: !torch.vtensor<[1,49,42,27],f32>, %arg1: !torch.vtensor<[3,3,27,28],f32>, %arg2: !torch.vtensor<[28],f32>) -> !torch.vtensor<[1,45,40,28],f32> {
    %int2 = torch.constant.int 2 
    %int1 = torch.constant.int 1 
    %int0 = torch.constant.int 0 
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %3 = torch.prim.ListConstruct %int0, %int0,%int0, %int0 : (!torch.int, !torch.int,!torch.int, !torch.int) -> !torch.list<!torch.int>
    %4 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %5 = "xten.conv2d"(%arg0, %arg1, %arg2, %2, %3, %4, %int1) : (!torch.vtensor<[1,49,42,27],f32>, !torch.vtensor<[3,3,27,28],f32>, !torch.vtensor<[28],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.vtensor<[1,45,40,28],f32>
    return %5 : !torch.vtensor<[1,45,40,28],f32>
  }
}