//===- aten_conv2d_add_relu.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
// CHECK-LABEL:  func.func @forward
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,8,8],f32>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[C2D:.*]] = "xten.conv2d"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST1]], %[[LIST1]], %int1) {layer_name = "Conv_1"} :  (!torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,2,8,8],f32>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_relu"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST1]], %[[LIST1]], %int1, %[[C2D]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,8,8],f32>) -> !torch.vtensor<[1,2,8,8],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,8,8],f32>
module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[1,2,8,8],f32>) -> !torch.vtensor<[1,2,8,8],f32> {
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %false = torch.constant.bool false
    %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<2x2x3x3xf32>) : !torch.vtensor<[2,2,3,3],f32>
    %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>

    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %c2d = torch.aten.convolution %arg0, %0, %none, %list1, %list1, %list1, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,8,8],f32>
    %c2d2 = torch.aten.convolution %arg0, %0, %none, %list1, %list1, %list1, %false, %empty_list, %int1 {layer_name = "Conv_1"} : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,8,8],f32>
    %add = torch.aten.add.Tensor %c2d, %c2d2, %int1 {layer_name = "Add_0"} : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[1,2,8,8],f32>, !torch.int ->  !torch.vtensor<[1,2,8,8],f32>
    %relu = torch.aten.relu %add {layer_name = "Relu_0"} : !torch.vtensor<[1,2,8,8],f32> -> !torch.vtensor<[1,2,8,8],f32>
    return %relu : !torch.vtensor<[1,2,8,8],f32>
  }
}