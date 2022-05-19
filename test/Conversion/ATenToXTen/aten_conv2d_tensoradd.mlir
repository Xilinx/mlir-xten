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


// --- conv2d_tensoradd_lrelu

// CHECK-LABEL:  func @forward_conv2d_tensoradd_lrelu
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_lrelu"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST1]], %[[LIST1]], %int1, %float4.000000e-01, %[[INPUT]]) : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func @forward_conv2d_tensoradd_lrelu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int1 = torch.constant.int 1
  %alpha = torch.constant.float 0.4
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>

  %c2d = torch.aten.conv2d %arg0, %0, %none, %list1, %list1, %list1, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  %lrelu = torch.aten.leaky_relu %add, %alpha : !torch.vtensor<[1,2,128,128],f32>, !torch.float -> !torch.vtensor<[1,2,128,128],f32>
  return %lrelu : !torch.vtensor<[1,2,128,128],f32>
}

// --- conv2d_tensoradd_relu

// CHECK-LABEL:  func @forward_conv2d_tensoradd_relu
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_relu"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST1]], %[[LIST1]], %int1, %[[INPUT]]) : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func @forward_conv2d_tensoradd_relu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %c2d = torch.aten.conv2d %arg0, %0, %none, %list1, %list1, %list1, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  %relu = torch.aten.relu %add : !torch.vtensor<[1,2,128,128],f32> -> !torch.vtensor<[1,2,128,128],f32>
  return %relu : !torch.vtensor<[1,2,128,128],f32>
}

// --- conv2d_tensoradd

// CHECK-LABEL:  func @forward_conv2d_tensoradd
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST1]], %[[LIST1]], %int1, %[[INPUT]]) : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,128,128],f32>
func @forward_conv2d_tensoradd(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>
  %list1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %c2d = torch.aten.conv2d %arg0, %0, %none, %list1, %list1, %list1, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
  %add = torch.aten.add.Tensor %c2d, %arg0, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
  return %add : !torch.vtensor<[1,2,128,128],f32>
}
