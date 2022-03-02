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

module attributes {torch.debug_module_name = "TinyYoloV2"}  {
  func @forward(%arg0: !torch.vtensor<[1,3,416,416],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %0 = torch.vtensor.literal(dense<0.0> : tensor<16x3x3x3xf32>) : !torch.vtensor<[16,3,3,3],f32>
    %1 = torch.vtensor.literal(dense<[0.172495767, 0.0930103883, -0.108694486, 0.164058045, 0.0867542847, 0.0829808115, 4.64801633E-5, 0.0927615389, 0.0260046888, -0.0928559899, 0.0503488705, -0.188906699, 0.0769577771, -0.143801734, 0.0231449194, 0.0710446536]> : tensor<16xf32>) : !torch.vtensor<[16],f32>
    %2 = torch.vtensor.literal(dense<0.0> : tensor<75x1024x1x1xf32>) : !torch.vtensor<[75,1024,1,1],f32>

    %3 = torch.vtensor.literal(dense<[0.025605306, -0.00545778126, -0.0174233839, 0.0210985467, 0.00263799354, -0.00916593894, 0.0195029676, -0.0131081864, 5.0765276E-4, 0.00126140192, -0.0197998062, -0.0281544328, -0.0181499757, -0.0283667557, -0.00685672089, 0.02513282, 0.00770622119, -0.0047609359, 0.0252044648, 0.0168272965, -0.00112291425, -2.036570e-02, -0.0259842761, -0.0202160627, -0.00638639926, -0.00904661789, -0.00340657309, -0.0271910094, -0.00292545557, -0.0263917297, 0.0108150318, 0.00230274722, -0.00291414931, 0.0215295181, 0.00143703073, -0.020694539, -0.0225435086, 0.0276760384, -2.920520e-02, 0.0181033462, -0.0081120953, -0.0140063614, -0.0209200457, -0.0285220593, -0.0254359245, -0.0312482156, -1.57617033E-4, -0.013407506, -0.020228073, 1.71646476E-4, 0.020688951, 0.00905153527, -0.00244066864, -0.0057259649, -0.00833169371, 0.0220405571, -0.0169333592, -0.0182871856, -0.00661813468, 0.0114165992, 0.0207470469, -0.0121288821, -0.0282617547, 0.0160249397, 0.00793497264, -0.0128190964, -0.0251259729, -0.0157120936, -0.0294440277, 0.0270810798, -0.021799624, 0.0305884443, -0.00690260902, 0.00997266918, 0.0261750072]> : tensor<75xf32>) : !torch.vtensor<[75],f32>
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %5 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %6 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %7 = torch.aten.conv2d %arg0, %0, %1, %4, %5, %6, %int1 {layer_name = "conv2d0"} : !torch.vtensor<[1,3,416,416],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
    %8 = torch.aten.leaky_relu %7, %float1.000000e-01 {layer_name = "leaky_relu0"} : !torch.vtensor<[?,?,?,?],f32>, !torch.float -> !torch.vtensor
    %9 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %10 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %11 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %12 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %13 = torch.aten.max_pool2d %8, %9, %10, %11, %12, %false {layer_name = "max_pool2d0"} : !torch.vtensor, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.bool -> !torch.vtensor<[?,?,?,?],unk>
    %14 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %15 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %16 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %17 = torch.aten.conv2d %13, %2, %3, %14, %15, %16, %int1 {layer_name = "conv2d1"} : !torch.vtensor<[?,?,?,?],unk>, !torch.vtensor<[75,1024,1,1],f32>, !torch.vtensor<[75],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
    return %17 : !torch.vtensor<[?,?,?,?],f32>
  }
}

