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

func.func @valid_lowering(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>, %arg2: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %0 = torch.aten.linear %arg0, %arg1, %arg2 {layer_name = "Linear_0"} : !torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
    
// CHECK-LABEL: func.func @valid_lowering
// CHECK-NEXT:     [[R0:%.+]] = "xten.linear"(%arg0, %arg1, %arg2) {layer_name = "Linear_0"} : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:     return [[R0]] : !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:  }
}


// We currently do not support a 3D input tensor. This should not be lowered to xten linear
func.func @invalid_lowering(%arg0: !torch.vtensor<[3,4,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>, %arg2: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[3, 4, 1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[3, 4, 1000],f32>
    return %0 : !torch.vtensor<[3, 4, 1000],f32>
    
// CHECK-LABEL: func.func @invalid_lowering
// CHECK-NEXT:     %[[R0:.+]] =  torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[3,4,1000],f32>
// CHECK-NEXT:     return %[[R0]] : !torch.vtensor<[3,4,1000],f32>
// CHECK-NEXT:  }
}

// If we receive an optional bias tensor we should just pass it along
func.func @lowering_with_bias_optional(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>) -> !torch.vtensor<[1,1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %none = torch.constant.none
    %optional_tensor = torch.derefine %none: !torch.none to !torch.optional<tensor>
    %0 = torch.aten.linear %arg0, %arg1, %optional_tensor : !torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.optional<tensor> -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
    
// CHECK-LABEL: func.func @lowering_with_bias_optional
// CHECK-NEXT:     %[[NONE:.+]] = torch.constant.none
// CHECK-NEXT:     %[[OPTIONAL:.+]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tensor>
// CHECK-NEXT:     %[[R0:.+]] = "xten.linear"(%arg0, %arg1, %[[OPTIONAL]]) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.optional<tensor>) -> !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:     return %[[R0]] : !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:  }
}

// If we receive an none as a bias tensor we should just pass it along, similar to the other test cases
func.func @lowering_with_bias_none(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>) -> !torch.vtensor<[1,1000],f32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
    %none = torch.constant.none
    %0 = torch.aten.linear %arg0, %arg1, %none : !torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.none -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
    
// CHECK-LABEL: func.func @lowering_with_bias_none
// CHECK-NEXT:     %[[NONE:.+]] = torch.constant.none
// CHECK-NEXT:     %[[R0:.+]] = "xten.linear"(%arg0, %arg1, %[[NONE]]) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.none) -> !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:     return %[[R0]] : !torch.vtensor<[1,1000],f32>
// CHECK-NEXT:  }
}
