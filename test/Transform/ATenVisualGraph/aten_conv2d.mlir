//===- aten_conv2d.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-visual-graph='operators-supported-path=%S/../../../lib/Transform/operators_supported.json' | FileCheck %s
// CHECK-LABEL:     "{{.*}}": {
// CHECK-LABEL:     "name": "conv2d0",
// CHECK-LABEL:     "name": "torch.aten.conv2d",
//

module attributes {torch.debug_module_name = "model"}  {
  func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.vtensor.literal(dense<0.0> : tensor<16x2x3x3xf32>) : !torch.vtensor<[16,2,3,3],f32>
    %1 = torch.vtensor.literal(dense<[0.132059276, -0.0918224751, -0.118777044, 0.0645219385, 0.134561762, -0.04097775, 0.182373062, -0.113158949, -0.0643238425, -0.0199289974, -0.073821865, -0.202036336, 0.149756551, -0.202734962, 0.169865787, -0.135248795]> : tensor<16xf32>) : !torch.vtensor<[16],f32>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %5 = torch.aten.conv2d %arg0, %0, %1, %2, %3, %4, %int1 {layer_name = "conv2d0"} : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?,?],f32>
  }
}
