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
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_lrelu_globalaveragepool"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %float4.000000e-01, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,1,1],f32>
func.func @forward_conv2d_tensoradd_lrelu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32> {
  %0 = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %none = torch.constant.none
  %float4.000000e-01 = torch.constant.float 4.000000e-01
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = "xten.conv2d_tensoradd_lrelu"(%arg0, %0, %none, %2, %1, %2, %int1, %float4.000000e-01, %arg0) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
  %4 = "xten.globalaveragepool2d"(%3) {layer_name = "GlobalAveragePool_0"} : (!torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
  return %4 : !torch.vtensor<[1,2,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @forward_conv2d_tensoradd_relu
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST0:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_relu_globalaveragepool"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,1,1],f32>
func.func @forward_conv2d_tensoradd_relu(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32> {
  %0 = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = "xten.conv2d_tensoradd_relu"(%arg0, %0, %none, %2, %1, %2, %int1, %arg0) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
  %4 = "xten.globalaveragepool2d"(%3) {layer_name = "GlobalAveragePool_0"} : (!torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
  return %4 : !torch.vtensor<[1,2,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @forward_conv2d_tensoradd
// CHECK-SAME:                 %[[INPUT:.*]]: !torch.vtensor<[1,2,128,128],f32>
// CHECK:   %[[LIST0:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[LIST1:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:   %[[OUT:.*]] = "xten.conv2d_tensoradd_globalaveragepool"(%[[INPUT]], %0, %none, %[[LIST1]], %[[LIST0]], %[[LIST1]], %int1, %[[INPUT]]) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
// CHECK-NEXT: return %[[OUT]] : !torch.vtensor<[1,2,1,1],f32>
func.func @forward_conv2d_tensoradd(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32> {
  %0 = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<2x2x1x1xf32>) : !torch.vtensor<[2,2,1,1],f32>
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = "xten.conv2d_tensoradd"(%arg0, %0, %none, %2, %1, %2, %int1, %arg0) {layer_name = "Conv_0"} : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
  %4 = "xten.globalaveragepool2d"(%3) {layer_name = "GlobalAveragePool_0"} : (!torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,1,1],f32>
  return %4 : !torch.vtensor<[1,2,1,1],f32>
}
