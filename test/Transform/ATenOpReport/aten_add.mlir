//===- aten_add.mlir -------------------------------------------*- MLIR -*-===//
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
//   CHECK-LABEL:     "{{.*}}": {
//   CHECK-NEXT:        "activation_in": 12,
//   CHECK-NEXT:        "activation_out": 6,
//   CHECK-NEXT:        "ops:+": 6,
//   CHECK-NEXT:        "reads": 12,
//   CHECK-NEXT:        "writes": 6
func.func @forward(%arg0: !torch.vtensor<[1,2,3],f32>, %arg1: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
