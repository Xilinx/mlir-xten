//===- xten_to_linalg_conv2d_relu.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK:   func.func @forward(%[[ARG:.*]]: {{.*}})
// CHECK-NEXT:     %[[CST0:.*]] = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<4096x25088xf32>)
// CHECK-NEXT:     %[[CST1:.*]] = torch.vtensor.literal(dense<-1.18024689E+29> : tensor<4096xf32>) 
// CHECK-NEXT:     %[[ARG_CTENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : {{.*}}
// CHECK-NEXT:     %[[CST0_CTENSOR:.*]] = torch_c.to_builtin_tensor %[[CST0]] : {{.*}} 
// CHECK-NEXT:     %[[CST1_CTENSOR:.*]] = torch_c.to_builtin_tensor %[[CST1]] : {{.*}}
// CHECK-NEXT:     %[[OUT:.*]] = tensor.empty() 
// CHECK-NEXT:     %[[LINEAR_RELU:.*]] = linalg.linear_relu {layer_name = "Gemm_0"} ins(%[[ARG_CTENSOR]], %[[CST0_CTENSOR]], %[[CST1_CTENSOR]] : {{.*}}) outs(%[[OUT]] : {{.*}})
// CHECK-NEXT:     %[[RES:.*]] = torch_c.from_builtin_tensor %[[LINEAR_RELU]] : tensor<1x4096xf32>
// CHECK-NEXT:     return %[[RES]] : !torch.vtensor<[1,4096],f32>

module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: !torch.vtensor<[1,25088],f32>) -> !torch.vtensor<[1,4096],f32> {
    %0 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<4096x25088xf32>) : !torch.vtensor<[4096,25088],f32>
    %1 = torch.vtensor.literal(dense<"0xDEADBEEF"> : tensor<4096xf32>) : !torch.vtensor<[4096],f32>
    %2 = "xten.linear_relu"(%arg0, %0, %1) {layer_name = "Gemm_0"} : (!torch.vtensor<[1,25088],f32>, !torch.vtensor<[4096,25088],f32>, !torch.vtensor<[4096],f32>) -> !torch.vtensor<[1,4096],f32>
    return %2 : !torch.vtensor<[1,4096],f32>
  }
}