//===- aten_conv2d_nobias.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
// CHECK: %[[INT1:.*]] = torch.constant.int 1
// CHECK: %[[INT2:.*]] = torch.constant.int 2
// CHECK: %[[INT1L:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[INT2L:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[C2D:.*]] = "xten.conv2d"(%[[IN1:.*]], %0, %none, %[[INT1L]], %[[INT1L]], %[[INT2L]], %[[INT1]]) : (!torch.vtensor<[1,2,256,256],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,2,128,128],f32>
// CHECK: %[[OUT:.*]] = "xten.conv2d_tensoradd_lrelu"(%[[IN2:.*]], %0, %none, %[[INT1L]], %[[INT1L]], %[[INT1L]], %[[INT1]], %float4.000000e-01, %[[C2D]]) : (!torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.float, !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32>
module attributes {torch.debug_module_name = "model"}  {
  func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[1,2,128,128],f32> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %alpha = torch.constant.float 0.4
    %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>
    %none = torch.constant.none
    %int1List = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %stride2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>

    %largeTensor = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<1x2x256x256xf32>) : !torch.vtensor<[1,2,256,256],f32>
    // this one gets fused
    %c2d1 = torch.aten.conv2d %arg0, %0, %none, %int1List, %int1List, %int1List, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>
    // not this one
    %c2d2 = torch.aten.conv2d %largeTensor, %0, %none, %int1List, %int1List, %stride2, %int1 : !torch.vtensor<[1,2,256,256],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,128,128],f32>

    %7 = torch.aten.add.Tensor %c2d2, %c2d1, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[1,2,128,128],f32>, !torch.int ->  !torch.vtensor<[1,2,128,128],f32>
    %8 = torch.aten.leaky_relu %7, %alpha : !torch.vtensor<[1,2,128,128],f32>, !torch.float -> !torch.vtensor<[1,2,128,128],f32>
    return %8 : !torch.vtensor<[1,2,128,128],f32>
  }
}
