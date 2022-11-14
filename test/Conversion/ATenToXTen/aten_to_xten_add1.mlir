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

// CHECK: "xten.add"(%{{.*}}, %{{.*}}) {layer_name = "Add_0"} : (!torch.vtensor<[256,256],f32>, !torch.vtensor<[256,256],f32>) -> !torch.vtensor<[256,256],f32>
module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[256,256],f32>, %arg1: !torch.vtensor<[256,256],f32>) -> !torch.vtensor<[256,256],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 {layer_name = "Add_0"} : !torch.vtensor<[256,256],f32>, !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f32>
    return %0 : !torch.vtensor<[256,256],f32>
  }
}