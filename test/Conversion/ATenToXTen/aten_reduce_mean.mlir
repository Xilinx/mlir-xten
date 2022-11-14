//===------------------------- aten_reduce_mean.mlir ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s

func.func @test_reduce_mean(%arg0: !torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %list = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.mean.dim %arg0, %list, %true, %none {layer_name = "mean.dim0"} : !torch.vtensor<[1,2048,7,7],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2048,1,1],f32>
    return %0 : !torch.vtensor<[1,2048,1,1],f32>

// CHECK-LABEL: func.func @test_reduce_mean
// CHECK-SAME:  ([[PARAM_0_:%.+]]: !torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = "xten.globalaveragepool2d"([[PARAM_0_]]) {layer_name = "mean.dim0"} : (!torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32>
// CHECK-NEXT:     return [[VAR_0_]] : !torch.vtensor<[1,2048,1,1],f32>
// CHECK-NEXT:  }
}
