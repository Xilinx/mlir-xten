//===- tiny_yolo_v2_block.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-visual-graph='operators-supported-path=%S/../../../lib/Transform/operators_supported.json' | FileCheck %s
// CHECK-LABEL:     "{{.*}}": {
// CHECK-LABEL:     "connections"
// CHECK:            {
// CHECK-NEXT:       "from_port_id": "{{[0-9]+}}", 
// CHECK-NEXT:       "id": "{{[0-9]+}}",
// CHECK-NEXT:       "to_port_id": "{{[0-9]+}}"
// CHECK-NEXT:       },
// CHECK:            {
// CHECK-NEXT:       "from_port_id": "{{[0-9]+}}", 
// CHECK-NEXT:       "id": "{{[0-9]+}}",
// CHECK-NEXT:       "to_port_id": "{{[0-9]+}}"
// CHECK-NEXT:       },
// CHECK:            {
// CHECK-NEXT:       "from_port_id": "{{[0-9]+}}", 
// CHECK-NEXT:       "id": "{{[0-9]+}}",
// CHECK-NEXT:       "to_port_id": "{{[0-9]+}}"
// CHECK-NEXT:       }
// CHECK-LABEL:     "design_name": "design 1",

module attributes {torch.debug_module_name = "TinyYoloV2"} {
  func @forward(%arg0: !torch.vtensor<[1,256,13,13],f32>) -> !torch.vtensor<[1,512,13,13],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<512x256x3x3xf32>) : !torch.vtensor<[512,256,3,3],f32>
    %4 = torch.vtensor.literal(dense<0.000000e+00> : tensor<512xf32>) : !torch.vtensor<[512],f32>
    %5 = torch.aten.conv2d %arg0, %3, %4, %1, %1, %1, %int1 {layer_name = "conv2d5"} : !torch.vtensor<[1,256,13,13],f32>, !torch.vtensor<[512,256,3,3],f32>, !torch.vtensor<[512],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,512,13,13],f32>
    %6 = torch.aten.leaky_relu %5, %float1.000000e-01 {layer_name = "leaky_relu5"} : !torch.vtensor<[1,512,13,13],f32>, !torch.float -> !torch.vtensor<[1,512,13,13],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %7 = torch.prim.ListConstruct %int0, %int1, %int0, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %8 = torch.aten.constant_pad_nd %6, %7, %float-Inf {layer_name = "constant_pad_nd0"} : !torch.vtensor<[1,512,13,13],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,512,14,14],f32>
    %9 = torch.aten.max_pool2d %8, %2, %1, %0, %1, %false {layer_name = "max_pool2d5"} : !torch.vtensor<[1,512,14,14],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,512,13,13],f32>
    return %9 : !torch.vtensor<[1,512,13,13],f32>
  }
}

