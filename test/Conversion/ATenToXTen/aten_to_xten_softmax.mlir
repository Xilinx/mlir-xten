//===- aten_to_xten_softmax.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This is the torch-mlir output for a module with a single softmax operator
// the decomposition pass has been disabled.

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
module attributes {torch.debug_module_name = "Model"} {
// CHECK: func.func @forward(%[[INPUT:.*]]: !torch.vtensor
  func.func @forward(%arg0: !torch.vtensor<[1,3,128,128],f32>) -> !torch.vtensor<[1,3,128,128],f32> {
    // CHECK: %[[DIM:.*]] = torch.constant.int -1
    %int-1 = torch.constant.int -1
    // CHECK: %[[HALF_TO_FLOAT:.*]] = torch.constant.bool
    %false = torch.constant.bool false
    // CHECK: %[[SOFTMAX:.*]] = "xten.softmax"(%[[INPUT]], %[[DIM]], %[[HALF_TO_FLOAT]]) {layer_name = "Softmax_0"} : (!torch.vtensor<[1,3,128,128],f32>, !torch.int, !torch.bool) -> !torch.vtensor<[1,3,128,128],f32>
    %0 = torch.aten._softmax %arg0, %int-1, %false {layer_name = "Softmax_0"} : !torch.vtensor<[1,3,128,128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,3,128,128],f32>
    return %0 : !torch.vtensor<[1,3,128,128],f32>
  }
}
