//===- aten_long_conv2d_add.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s

// Verify that the conv2d with the longest path to a common ancestor is fused
// with the subsequent add.
//
// CHECK: %[[COMMON:.*]] = "xten.add"(%arg0, %arg1
// CHECK: %[[C2D1:.*]] = "xten.conv2d"(%[[COMMON]]
// CHECK: %[[C2D1a:.*]] = "xten.conv2d"(%[[COMMON]]
// CHECK: %[[OUT:.*]] = "xten.conv2d_tensoradd_relu"(%[[C2D1]]
// CHECK-SAME: %[[C2D1a]])

module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>, %arg1: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %weight = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<2x2x3x3xf32>) : !torch.vtensor<[2,2,3,3],f32>
    %none = torch.constant.none
    %int1List = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>

    %common = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
    %c2d1 = torch.aten.conv2d %common, %weight, %none, %int1List, %int1List, %int1List, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
    %c2d1a = torch.aten.conv2d %common, %weight, %none, %int1List, %int1List, %int1List, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
    %c2d2 = torch.aten.conv2d %c2d1, %weight, %none, %int1List, %int1List, %int1List, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
    %7 = torch.aten.add.Tensor %c2d2, %c2d1a, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
    %8 = torch.aten.relu %7 : !torch.vtensor<[1,2,128,128],f32> -> !torch.vtensor<[1,2,128,128],f32>
    return %8 : !torch.vtensor<[1,2,128,128],f32>
  }
}
