//===- .mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt "%s" -linalg-to-dse -o /dev/null 2>&1 | FileCheck "%s"
// CHECK: error: unmatched generic broadcast operator
#map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "HelloWorld"}  {
  func @forward(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x125x7x7xf32> {
    %cst_17 = arith.constant dense<1.000000e+00> : tensor<125xf32>
    %23 = linalg.init_tensor [1, 125, 7, 7] : tensor<1x125x7x7xf32>
    %24 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_17 : tensor<125xf32>) outs(%23 : tensor<1x125x7x7xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x125x7x7xf32>
    return %24 : tensor<1x125x7x7xf32>
  }
}
