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
// CHECK: linalg.globalaveragepool2d ins({{.*}} : tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>

func.func @test_reduce_mean(%arg0: !torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32> {
    %0 = "xten.globalaveragepool2d"(%arg0) : (!torch.vtensor<[1,2048,7,7],f32>) -> !torch.vtensor<[1,2048,1,1],f32>
    return %0 : !torch.vtensor<[1,2048,1,1],f32>
}
