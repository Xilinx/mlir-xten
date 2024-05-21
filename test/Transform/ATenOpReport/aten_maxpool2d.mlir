//===- aten_maxpool2d.mlir -------------------------------------*- MLIR -*-===//
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
//   CHECK-LABEL:  "{{.*}}": {
//   CHECK-NEXT:    "activation_in": 8192,
//   CHECK-NEXT:    "activation_out": 2048,
//   CHECK-NEXT:    "ops:>": 6144,
//   CHECK-NEXT:    "reads": 8192,
//   CHECK-NEXT:    "writes": 2048

module attributes {torch.debug_module_name = "max_pool2d"} {
  func.func @forward(%arg0: !torch.vtensor<[1,32,16,16],f32>) -> !torch.vtensor<[1,32,8,8],f32> {
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.max_pool2d %arg0, %0, %0, %1, %2, %false : !torch.vtensor<[1,32,16,16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,32,8,8],f32>
    return %3 : !torch.vtensor<[1,32,8,8],f32>
  }
}
