//===- aten_addmm.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// XFAIL: *
// RUN: aten-opt %s --aten-op-report | FileCheck %s
//   CHECK-LABEL:     "unknown-layer-2": {
//   CHECK-NEXT:        "activation_in": 1024,
//   CHECK-NEXT:        "activation_out": 16,
//   CHECK-NEXT:        "ops:+": 16,
//   CHECK-NEXT:        "ops:MAC": 16384,
//   CHECK-NEXT:        "parameters_in": 16400,
//   CHECK-NEXT:        "reads": 17424,
//   CHECK-NEXT:        "writes": 16

module attributes {torch.debug_module_name = "addmm"} {
  func @forward(%arg0: !torch.vtensor<[1,1024],f32>, %arg1: !torch.vtensor<[16,1024],f32>, %arg2: !torch.vtensor<[1,16],f32>, %arg3: !torch.vtensor<[1,16],f32>) -> !torch.vtensor<[1,16],f32> {
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[16,1024],f32>, !torch.int, !torch.int -> !torch.vtensor<[1024,16],f32>
    %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[1,1024],f32>, !torch.vtensor<[1024,16],f32> -> !torch.vtensor<[1,16],f32>
    %2 = torch.aten.mul.Scalar %arg2, %int2 : !torch.vtensor<[1,16],f32>, !torch.int -> !torch.vtensor<[1,16],f32>
    %3 = torch.aten.add.Tensor %2, %1, %int2 : !torch.vtensor<[1,16],f32>, !torch.vtensor<[1,16],f32>, !torch.int -> !torch.vtensor<[1,16],f32>
    return %3 : !torch.vtensor<[1,16],f32>
  }
}