//===- aten_add.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-op-report | FileCheck %s
//   CHECK-LABEL:     "{{.*}}": {
//   CHECK-NEXT:        "activation_in": 12,
//   CHECK-NEXT:        "activation_out": 6,
//   CHECK-NEXT:        "ops:+": 6,
//   CHECK-NEXT:        "reads": 12,
//   CHECK-NEXT:        "writes": 6
func @graph(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %c1_i64 = constant 1 : i64
  %0 = "aten.add"(%arg0, %arg1, %c1_i64) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, i64) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
