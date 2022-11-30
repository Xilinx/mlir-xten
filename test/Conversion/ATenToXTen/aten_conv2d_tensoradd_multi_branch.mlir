//===- aten_conv2d_tensoradd_multi_branch.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s


// CHECK: %[[CONV1:.*]] = "xten.conv2d"(%[[ARG0:.*]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_0"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d_tensoradd"(%[[ARG0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ARG0]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV2]]) {layer_name = "Conv_1"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func.func @conv2d_tensoradd_multi_branch(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
    %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
    %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %4 = torch.aten.convolution %arg0, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
    %5 = torch.aten.convolution %4, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_1"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
    %6 = torch.aten.add.Tensor %arg0, %4, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
    %7 = torch.aten.add.Tensor %6, %5, %int1 {layer_name = "Add_1"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
    %8 = torch.aten.relu %7 {layer_name = "Relu_0"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
    return %8 : !torch.vtensor<[1,256,64,64],f32>
}
