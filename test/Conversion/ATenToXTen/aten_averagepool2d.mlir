//===------------------ aten_averagepool2d.mlir -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten | FileCheck %s

// Torch may also represent a reducemean as an average pool 2D with specific settings. Below is an
// example of those settings and semantics.
func.func @test_averagepool_2d_7x7(%arg0: !torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int7 = torch.constant.int 7
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
    return %3 : !torch.vtensor<[1,512,1,1],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_7x7
// CHECK:         %[[RES:.*]] = "xten.globalaveragepool2d"(%arg0) {layer_name = "avg_pool2d0"} : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,1,1],f32>
// CHECK-NEXT:    return %[[RES]] : !torch.vtensor<[1,512,1,1],f32>
// CHECK-NEXT:  }
}

func.func @test_averagepool_2d_3x3(%arg0: !torch.vtensor<[1,512,3,3],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
    return %3 : !torch.vtensor<[1,512,1,1],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_3x3
// CHECK:         %[[RES:.*]] = "xten.globalaveragepool2d"(%arg0) {layer_name = "avg_pool2d0"} : (!torch.vtensor<[1,512,3,3],f32>) -> !torch.vtensor<[1,512,1,1],f32>
// CHECK-NEXT:    return %[[RES]] : !torch.vtensor<[1,512,1,1],f32>
// CHECK-NEXT:  }
}

// ifm is 7x7 but the kernel size is 3x3 so it is not global
func.func @test_averagepool_2d_incorrect_arguments(%arg0: !torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,4,4],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,4,4],f32>
    return %3 : !torch.vtensor<[1,512,4,4],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_incorrect_arguments
// CHECK:         %[[RES:.*]] = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} 
}

// We only convert if the stride is [1,1] here it is [2,2]
func.func @test_averagepool_2d_larger_stride(%arg0: !torch.vtensor<[1,512,3,3],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
    return %3 : !torch.vtensor<[1,512,1,1],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_larger_stride
// CHECK:         %[[RES:.*]] = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %none {layer_name = "avg_pool2d0"} 
}

// We only convert if ceil_mode is false
func.func @test_averagepool_2d_ceil_mode_true(%arg0: !torch.vtensor<[1,512,3,3],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %true, %true, %none {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
    return %3 : !torch.vtensor<[1,512,1,1],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_ceil_mode_true
// CHECK:         %[[RES:.*]] = torch.aten.avg_pool2d %arg0, %2, %0, %1, %true, %true, %none {layer_name = "avg_pool2d0"}
}

// We only convert if the divisor override is none, here it is set to 1
func.func @test_averagepool_2d_divisor_one(%arg0: !torch.vtensor<[1,512,3,3],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %int1 {layer_name = "avg_pool2d0"} : !torch.vtensor<[1,512,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.int -> !torch.vtensor<[1,512,1,1],f32>
    return %3 : !torch.vtensor<[1,512,1,1],f32>

// CHECK-LABEL: func.func @test_averagepool_2d_divisor_one
// CHECK:         %[[RES:.*]] = torch.aten.avg_pool2d %arg0, %2, %0, %1, %false, %true, %int1 {layer_name = "avg_pool2d0"}
}