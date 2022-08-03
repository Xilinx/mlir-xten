//===- aten_linear_to_xten.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s

func @valid_lowering(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>, %arg2: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
    
// CHECK-LABEL: func @valid_lowering
// CHECK-NEXT:     [[R0:%.+]] = "xten.linear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:     return [[R0]] : !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:  }
}


// We currently do not support a 3D input tensor. This should not be lowered to xten linear
func @invalid_lowering(%arg0: !torch.vtensor<[3,4,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>, %arg2: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[3, 4, 1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[3, 4, 1000],f32>
    return %0 : !torch.vtensor<[3, 4, 1000],f32>
    
// CHECK-LABEL: func @invalid_lowering
// CHECK-NEXT:     %[[R0:.+]] =  torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[3,4,1000],f32>
// CHECK-NEXT:     return %[[R0]] : !torch.vtensor<[3,4,1000],f32>
// CHECK-NEXT:  }
}
