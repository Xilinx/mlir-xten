//===- xten_to_linalg_mm.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK: %[[OUT:.*]] = linalg.fill
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x32xf32>, tensor<32x64xf32>) outs(%[[OUT]] : tensor<1x64xf32>)
module  {
  func @myFunc(%arg0: !torch.vtensor<[1,32],f32>, %arg1: !torch.vtensor<[32,64],f32>) -> !torch.vtensor<[1,64],f32> {
    %0 = "xten.mm"(%arg0, %arg1) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[32,64],f32>) -> !torch.vtensor<[1,64],f32>
    return %0 : !torch.vtensor<[1,64],f32>
  }
}
