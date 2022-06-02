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
// CHECK: func @forward(%[[INPUT:.*]]: !torch.vtensor
  func @forward(%arg0: !torch.vtensor<[1,3,128,128],f32>) -> !torch.vtensor<[1,3,128,128],f32> {
    // CHECK: %[[DIM:.*]] = torch.constant.int -1
    %int-1 = torch.constant.int -1
    // CHECK: %[[DTYPE:.*]] = torch.constant.none
    %none = torch.constant.none
    // CHECK: %[[SOFTMAX:.*]] = "xten.softmax"(%[[INPUT]], %[[DIM]], %[[DTYPE]]) : (!torch.vtensor<[1,3,128,128],f32>, !torch.int, !torch.none) -> !torch.vtensor<[1,3,128,128],f32>
    %0 = torch.aten.softmax.int %arg0, %int-1, %none : !torch.vtensor<[1,3,128,128],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,3,128,128],f32>
    return %0 : !torch.vtensor<[1,3,128,128],f32>
  }
}
