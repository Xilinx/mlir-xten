//===- aten_resnet_block.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten -split-input-file | FileCheck %s


// --- conv2d_tensoradd_branch_relu

// CHECK-LABEL:  func.func @conv2d_tensoradd_branch_relu

// CHECK: %[[CONV1:.*]] = "xten.conv2d_relu"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_0"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVSKIP:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_1"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d_relu"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_2"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d_relu"(%[[CONV2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_3"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVADD:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONVSKIP]]) {layer_name = "Conv_4"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func.func @conv2d_tensoradd_branch_relu(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
  %4 = torch.aten.convolution %arg0, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %5 = torch.aten.relu %4 {layer_name = "Relu_0"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %6 = torch.aten.convolution %5, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_1"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %7 = torch.aten.convolution %5, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_2"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %8 = torch.aten.relu %7 {layer_name = "Relu_1"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %9 = torch.aten.convolution %8, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_3"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %10 = torch.aten.relu %9 {layer_name = "Relu_2"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %11 = torch.aten.convolution %10, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_4"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %12 = torch.aten.add.Tensor %6, %11, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %13 = torch.aten.relu %12 {layer_name = "Relu_3"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  return %13 : !torch.vtensor<[1,256,64,64],f32>
}

// --- conv2d_tensoradd_branch_relu_reversed

// CHECK-LABEL:  func.func @conv2d_tensoradd_branch_relu_reversed

// CHECK: %[[CONV1:.*]] = "xten.conv2d_relu"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_0"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVSKIP:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_1"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d_relu"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_2"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d_relu"(%[[CONV2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {layer_name = "Conv_3"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVADD:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONVSKIP]]) {layer_name = "Conv_4"} : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func.func @conv2d_tensoradd_branch_relu_reversed(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
  %4 = torch.aten.convolution %arg0, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %5 = torch.aten.relu %4 {layer_name = "Relu_0"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %6 = torch.aten.convolution %5, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_1"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %7 = torch.aten.convolution %5, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_2"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %8 = torch.aten.relu %7 {layer_name = "Relu_1"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %9 = torch.aten.convolution %8, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_3"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %10 = torch.aten.relu %9 {layer_name = "Relu_2"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %11 = torch.aten.convolution %10, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_4"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %12 = torch.aten.add.Tensor %11, %6, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %13 = torch.aten.relu %12 {layer_name = "Relu_3"} : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  return %13 : !torch.vtensor<[1,256,64,64],f32>
}
