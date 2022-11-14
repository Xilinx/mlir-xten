//===- aten_conv2d_relu_pad_maxpool2d.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
// CHECK: [[BIAS:%.]] = torch.vtensor.literal{{.*}} : !torch.vtensor<[64],f32>
// CHECK: [[WGTS:%.]] = torch.vtensor.literal{{.*}} !torch.vtensor<[64,3,7,7],f32>
// CHECK: [[LIST2:%.]] = torch.prim.ListConstruct %int2, %int2 :
// CHECK: [[LIST1:%.]] = torch.prim.ListConstruct %int1, %int1 :
// CHECK: [[LIST3:%.]] = torch.prim.ListConstruct %int3, %int3 :
// CHECK: [[LIST0101:%.]] = torch.prim.ListConstruct %int0, %int1, %int0, %int1 :
// CHECK: [[FUSED:%.]] = "xten.conv2d_relu_pad_maxpool"(%arg0, [[WGTS]], [[BIAS]], [[LIST2]], [[LIST3]], [[LIST1]], %int1, [[LIST0101]], %float-3.402820e38, %4, %2, %6, %3, %false) {layer_name = "Conv_0"} : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int>, !torch.float, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[1,64,56,56],f32>
// CHECK: return [[FUSED]] :
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,64,56,56],f32> {
    %float-3.402820e38 = torch.constant.float -3.4028234663852886E+38
    %false = torch.constant.bool false
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<0.000002> : tensor<64xf32>) : !torch.vtensor<[64],f32>
    %1 = torch.vtensor.literal(dense<1.000000e-07> : tensor<64x3x7x7xf32>) : !torch.vtensor<[64,3,7,7],f32>
    %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %5 = torch.aten.convolution %arg0, %1, %0, %2, %4, %3, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,112,112],f32>
    %6 = torch.aten.relu %5 {layer_name = "Relu_0"} : !torch.vtensor<[1,64,112,112],f32> -> !torch.vtensor<[1,64,112,112],f32>
    %7 = torch.prim.ListConstruct %int0, %int1, %int0, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %8 = torch.aten.constant_pad_nd %6, %7, %float-3.402820e38 {layer_name = "Pad_0"} : !torch.vtensor<[1,64,112,112],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,113,113],f32>
    %9 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %10 = torch.aten.max_pool2d %8, %4, %2, %9, %3, %false {layer_name = "Maxpool_0"} : !torch.vtensor<[1,64,113,113],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
    return %10 : !torch.vtensor<[1,64,56,56],f32>
  }
}
