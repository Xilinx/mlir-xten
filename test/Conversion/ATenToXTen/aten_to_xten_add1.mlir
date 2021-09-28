//===- aten_to_xten_add1.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --aten-to-xten | FileCheck %s

// CHECK: %[[VAL_1:.*]] = constant dense<1.111110e+00> : tensor<f32>
// CHECK: %[[VAL_2:.*]] = "xten.add_constant"(%{{.*}}, %[[VAL_1]]) : (tensor<256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
module {
  func @graph(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = constant dense<1.111110e+00> : tensor<f32>
    %c1_i64 = constant 1 : i64
    %0 = "aten.add"(%arg0, %cst, %c1_i64) : (tensor<256x256xf32>, tensor<f32>, i64) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
