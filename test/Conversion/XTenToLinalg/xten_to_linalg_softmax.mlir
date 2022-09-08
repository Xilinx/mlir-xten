//===- xten_to_linalg_softmax.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
module attributes {torch.debug_module_name = "Model"} {
  func.func @forward(%arg0: !torch.vtensor<[1,3,128,128],f32>) -> !torch.vtensor<[1,3,128,128],f32> {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    // CHECK: linalg.softmax {dim = -1 : i64} ins({{.*}} : tensor<1x3x128x128xf32>) -> tensor<1x3x128x128xf32>
    %0 = "xten.softmax"(%arg0, %int-1, %false) : (!torch.vtensor<[1,3,128,128],f32>, !torch.int, !torch.bool) -> !torch.vtensor<[1,3,128,128],f32>
    return %0 : !torch.vtensor<[1,3,128,128],f32>
  }
}