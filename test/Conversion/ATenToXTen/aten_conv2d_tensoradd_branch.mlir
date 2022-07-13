//===- aten_conv2d_tensoradd_branch.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten -split-input-file | FileCheck %s


// --- conv2d_tensoradd_branch

// CHECK-LABEL:  func @conv2d_tensoradd_branch

// CHECK: %[[CONV1:.*]] = "xten.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV4:.*]] = "xten.conv2d"(%[[CONV3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVADD:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV4]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV2]]) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func @conv2d_tensoradd_branch(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.conv2d %arg0, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %5 = torch.aten.conv2d %4, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %6 = torch.aten.conv2d %4, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %7 = torch.aten.conv2d %6, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %8 = torch.aten.conv2d %7, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %9 = torch.aten.add.Tensor %5, %8, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %10 = torch.aten.relu %9 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  return %10 : !torch.vtensor<[1,256,64,64],f32>
}

// --- conv2d_tensoradd_branch_reversed

// CHECK-LABEL:  func @conv2d_tensoradd_branch_reversed

// CHECK: %[[CONV1:.*]] = "xten.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV4:.*]] = "xten.conv2d"(%[[CONV3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVADD:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV4]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV2]]) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func @conv2d_tensoradd_branch_reversed(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.conv2d %arg0, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %5 = torch.aten.conv2d %4, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %6 = torch.aten.conv2d %4, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %7 = torch.aten.conv2d %6, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %8 = torch.aten.conv2d %7, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %9 = torch.aten.add.Tensor %8, %5, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %10 = torch.aten.relu %9 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  return %10 : !torch.vtensor<[1,256,64,64],f32>
}