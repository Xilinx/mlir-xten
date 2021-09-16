// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
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
