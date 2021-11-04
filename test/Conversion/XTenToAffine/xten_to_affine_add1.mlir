//===- air_to_affine_add1.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --xten-to-affine | FileCheck %s
// CHECK: affine.for %[[IV_1:.*]] = 0 to 4 {
// CHECK:   affine.for %[[IV_2:.*]] = 0 to 4 {
// CHECK:   %[[VAL_1:.*]] = affine.load %{{.*}}[%[[IV_1]], %[[IV_2]]] : memref<4x4xf32>
// CHECK:   %[[VAL_2:.*]] = constant 1.000000e+00 : f32
// CHECK:   %[[VAL_3:.*]] = addf %[[VAL_1]], %[[VAL_2]] : f32
// CHECK:   affine.store %[[VAL_3]], %{{.*}}[%[[IV_1]], %[[IV_2]]] : memref<4x4xf32>
module {
  func @graph(%arg0: !torch.vtensor<[4,4], f32>) -> !torch.vtensor<[4,4], f32> {
    %0 = torch.constant.float 1.0
    %1 = "xten.add_constant"(%arg0, %0) : (!torch.vtensor<[4,4], f32>, !torch.float) -> !torch.vtensor<[4,4], f32>
    return %1 : !torch.vtensor<[4,4], f32>
  }
}
