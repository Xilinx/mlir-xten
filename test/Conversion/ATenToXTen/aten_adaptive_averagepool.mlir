


//===------------------ aten_adaptive_average_pool.mlir -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s

func.func @test_adaptive_averagepool(%arg0: !torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32> {
    %int1 = torch.constant.int 1
    %list = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.adaptive_avg_pool2d %arg0, %list {layer_name = "adaptive_avg_pool2d0"} : !torch.vtensor<[1,2048,7,7],f32>, !torch.list<int> -> !torch.vtensor<[1,2048,1,1],f32>
    return %0 : !torch.vtensor<[1,2048,1,1],f32>

// CHECK-LABEL: func.func @test_adaptive_averagepool
// CHECK-SAME:  ([[PARAM_0_:%.+]]: !torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = "xten.globalaveragepool2d"([[PARAM_0_]]) {layer_name = "adaptive_avg_pool2d0"} : (!torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32>
// CHECK-NEXT:     return [[VAR_0_]] : !torch.vtensor<[1,2048,1,1],f32>
// CHECK-NEXT:  }
}
