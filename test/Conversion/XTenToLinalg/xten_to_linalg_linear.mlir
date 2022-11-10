//===--------------- xten_to_linalg_global_averagepool.mlir --------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-linalg | FileCheck %s

func.func @test_convert_linear(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>, %arg2: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32> {
    %0 = "xten.linear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
// CHECK: linalg.linear ins({{.+}} : tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) outs({{.+}} : tensor<1x1000xf32>) -> tensor<1x1000xf32>
}

func.func @test_convert_linear_bias_optional(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>) -> !torch.vtensor<[1,1000],f32> {
    %none = torch.constant.none
    %optional_tensor = torch.derefine %none: !torch.none to !torch.optional<tensor>
    %0 = "xten.linear"(%arg0, %arg1, %optional_tensor) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.optional<tensor>) -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1000xf32>
// CHECK-NEXT: %[[CONSTANT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[ZEROED:.+]] = linalg.fill ins(%[[CONSTANT]] : f32) outs(%[[EMPTY]] : tensor<1000xf32>) -> tensor<1000xf32>
// CHECK: linalg.linear ins({{.+}}, {{.+}}, %[[ZEROED]] : tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) outs({{.+}} : tensor<1x1000xf32>) -> tensor<1x1000xf32>
}

func.func @test_convert_linear_bias_none(%arg0: !torch.vtensor<[1,2048],f32>, %arg1: !torch.vtensor<[1000,2048],f32>) -> !torch.vtensor<[1,1000],f32> {
    %none = torch.constant.none
    %0 = "xten.linear"(%arg0, %arg1, %none) : (!torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.none) -> !torch.vtensor<[1,1000],f32>
    return %0 : !torch.vtensor<[1,1000],f32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1000xf32>
// CHECK-NEXT: %[[CONSTANT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[ZEROED:.+]] = linalg.fill ins(%[[CONSTANT]] : f32) outs(%[[EMPTY]] : tensor<1000xf32>) -> tensor<1000xf32>
// CHECK: linalg.linear ins({{.+}}, {{.+}}, %[[ZEROED]] : tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) outs({{.+}} : tensor<1x1000xf32>) -> tensor<1x1000xf32>
}