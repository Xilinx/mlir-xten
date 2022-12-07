//===- aten_linear_relu.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s
// CHECK:  func.func @forward(%[[ARG:.*]]: {{.*}})
// CHECK-NEXT:    %[[CST0:.*]] = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<4096x25088xf32>) 
// CHECK-NEXT:    %[[CST1:.*]] = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<4096xf32>)
// CHECK-NEXT:    %[[RES:.*]] = "xten.linear_relu"(%[[ARG]], %[[CST0]], %[[CST1]]) {layer_name = "Gemm_0"} 
// CHECK-NEXT:    return %[[RES]]

module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[1,25088],f32>) -> !torch.vtensor<[1,4096],f32> {
    %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<4096x25088xf32>) : !torch.vtensor<[4096,25088],f32>
    %1 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<4096xf32>) : !torch.vtensor<[4096],f32>
    %2 = torch.aten.linear %arg0, %0, %1 {layer_name = "Gemm_0"} : !torch.vtensor<[1,25088],f32>, !torch.vtensor<[4096,25088],f32>, !torch.vtensor<[4096],f32> -> !torch.vtensor<[1,4096],f32>
    %3 = torch.aten.relu %2 {layer_name = "Relu_0"} : !torch.vtensor<[1,4096],f32> -> !torch.vtensor<[1,4096],f32>
    return %3 : !torch.vtensor<[1,4096],f32>
  }
}