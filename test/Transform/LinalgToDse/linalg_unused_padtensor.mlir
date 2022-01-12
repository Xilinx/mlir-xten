//===- linalg_unused_padtensor.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt "%s" -linalg-to-dse -o /dev/null 2>&1 | FileCheck "%s"
// CHECK: error: unmatched pad operator
#map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "HelloWorld"}  {
  func @forward(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x66x66xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %3 = linalg.pad_tensor %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x16x64x64xf32> to tensor<1x16x66x66xf32>
    return %3 : tensor<1x16x66x66xf32>
  }
}
