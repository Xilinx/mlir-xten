//===- aten_conv2d_lrelu.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
// CHECK: [[WGTS:%.]] = torch.vtensor.literal{{.*}} !torch.vtensor<[16,2,1,1],f32>
// CHECK: [[BIAS:%.]] = torch.vtensor.literal{{.*}} : !torch.vtensor<[16],f32>
// CHECK: [[LIST1:%.]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[FUSED:%.]] = "xten.conv2d_lrelu"(%arg0, [[WGTS]], [[BIAS]], [[LIST1]], [[LIST1]], [[LIST1]], %int1, %float2.320000e-01) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,1,1],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float) -> !torch.vtensor<[1,16,130,130],f32>
// CHECK: return [[FUSED]] : !torch.vtensor<[1,16,130,130],f32>
module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,16,130,130],f32> {
    %int1 = torch.constant.int 1
    %alpha = torch.constant.float 0.232
    %false = torch.constant.bool false
    %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16x2x1x1xf32>) : !torch.vtensor<[16,2,1,1],f32>
    %bias = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16xf32>) : !torch.vtensor<[16],f32>
    %none = torch.constant.none
    %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %c2d = torch.aten.convolution %arg0, %0, %bias, %list1, %list1, %list1, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,1,1],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,130,130],f32>
    %lrelu = torch.aten.leaky_relu %c2d, %alpha {layer_name = "LeakyRelu_0"} : !torch.vtensor<[1,16,130,130],f32>, !torch.float -> !torch.vtensor<[1,16,130,130],f32>
    return %lrelu : !torch.vtensor<[1,16,130,130],f32>
  }
}