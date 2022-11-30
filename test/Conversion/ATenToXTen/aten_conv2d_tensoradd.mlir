//===- aten_conv2d_tensoradd.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten -split-input-file | FileCheck %s


// CHECK-LABEL:  func.func @forward_conv2d_tensoradd_lrelu
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST0:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_lrelu"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %float4.000000e-01, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func.func @forward_conv2d_tensoradd_lrelu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int0 = torch.constant.int 0
  %list0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %alpha = torch.constant.float 0.4
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>

  %c2d = torch.aten.convolution %arg0, %0, %none, %list1, %list0, %list1, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  %lrelu = torch.aten.leaky_relu %add, %alpha {layer_name = "LeakyRelu_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.float -> !torch.vtensor<[1,2,128,128],f32>
  return %lrelu : !torch.vtensor<[1,2,128,128],f32>
}

// -----

// CHECK-LABEL:  func.func @forward_conv2d_tensoradd_relu
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST0:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_relu"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func.func @forward_conv2d_tensoradd_relu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int0 = torch.constant.int 0
  %list0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
  %c2d = torch.aten.convolution %arg0, %0, %none, %list1, %list0, %list1, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  %relu = torch.aten.relu %add {layer_name = "Relu_0"} : !torch.vtensor<[1,2,128,128],f32> -> !torch.vtensor<[1,2,128,128],f32>
  return %relu : !torch.vtensor<[1,2,128,128],f32>
}

// -----

// CHECK-LABEL:  func.func @forward_conv2d_tensoradd
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST0:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func.func @forward_conv2d_tensoradd(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int0 = torch.constant.int 0
  %list0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
  %c2d = torch.aten.convolution %arg0, %0, %none, %list1, %list0, %list1, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  return %add : !torch.vtensor<[1,2,128,128],f32>
}
