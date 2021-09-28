//===- air_to_linalg_mm_tensors.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg='linalg-on-tensors=true' | FileCheck %s
// CHECK: %0 = linalg.init_tensor [128, 128] : tensor<128x128xi32>
// CHECK: %1 = linalg.matmul {{.*}} ins(%arg0, %arg1 : tensor<128x128xi32>, tensor<128x128xi32>) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
module  {
  func @task(%arg0: tensor<128x128xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = "xten.mm"(%arg0, %arg1) : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    return %0 : tensor<128x128xi32>
  }
}