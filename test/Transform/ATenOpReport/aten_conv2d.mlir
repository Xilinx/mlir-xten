//===- aten_conv2d.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2019 - 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --aten-op-report | FileCheck %s
//   CHECK-LABEL:   "{{.*}}": {
//   CHECK-NEXT:     "activation_in": 32768,
//   CHECK-NEXT:     "activation_out": 65536,
//   CHECK-NEXT:     "ops:+": 65536,
//   CHECK-NEXT:     "ops:MAC": 6422528,
//   CHECK-NEXT:     "parameters_in": 1584,
//   CHECK-NEXT:     "reads": 34352,
//   CHECK-NEXT:     "writes": 65536

module attributes {torch.debug_module_name = "conv_2d_7x7"} {
  func.func @forward(%arg0: !torch.vtensor<[1,2,128,128],f32>, %arg1: !torch.vtensor<[16,2,7,7],f32>, %arg2: !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,16,64,64],f32> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %5 = torch.aten.convolution %arg0, %arg1, %arg2, %2, %3, %4, %false, %empty_list, %int1 : !torch.vtensor<[1,2,128,128],f32>, !torch.vtensor<[16,2,7,7],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,64,64],f32>
    return %5 : !torch.vtensor<[1,16,64,64],f32>
  }
}
