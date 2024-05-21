//===- aten_batchnorm.mlir -------------------------------------*- MLIR -*-===//
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
//   CHECK-LABEL:       "{{.*}}": {
//   CHECK-NEXT:          "activation_in": 103320,
//   CHECK-NEXT:          "activation_out": 103320,
//   CHECK-NEXT:          "ops:*": 310206,
//   CHECK-NEXT:          "ops:+": 413280,
//   CHECK-NEXT:          "ops:-": 123,
//   CHECK-NEXT:          "ops:/": 123,
//   CHECK-NEXT:          "ops:sqrt": 123,
//   CHECK-NEXT:          "parameters_in": 246,
//   CHECK-NEXT:          "reads": 103566,
//   CHECK-NEXT:          "writes": 103320

module attributes {torch.debug_module_name = "batch_norm"} {
  func.func @forward(%arg0: !torch.vtensor<[42,123,4,5],f32>) -> !torch.vtensor<[42,123,4,5],f32> {
    %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<123xf32>) : !torch.vtensor<[123],f32>
    %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<123xf32>) : !torch.vtensor<[123],f32>
    %true = torch.constant.bool true
    %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    %4 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[42,123,4,5],f32>, !torch.vtensor<[123],f32>, !torch.vtensor<[123],f32>, !torch.vtensor<[123],f32>, !torch.vtensor<[123],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[42,123,4,5],f32>
    return %4 : !torch.vtensor<[42,123,4,5],f32>
  }
}
