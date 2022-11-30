//===- aten_to_xten_conv2d.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --aten-to-xten | FileCheck %s
// CHECK: %4 = "xten.conv2d"(%arg0, %1, %0, %2, %3, %2, %int1) {layer_name = "Conv_0"} : (!torch.vtensor<[1,1,128,128],f32>, !torch.vtensor<[8,1,3,3],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,8,126,126],f32>

module attributes {torch.debug_module_name = "conv2d"} {
  func.func @forward(%arg0: !torch.vtensor<[1,1,128,128],f32>) -> !torch.vtensor<[1,8,126,126],f32> {
    %0 = torch.vtensor.literal(dense<[-0.0488186777, -0.0371113718, -0.32024774, -0.187612861, 9.54508781E-4, -0.3219558, -0.297136664, 0.0279702246]> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %1 = torch.vtensor.literal(dense<[[[[0.00302931666, 0.242982954, 0.0806937515], [0.250730604, 0.0516703427, 0.178943545], [-0.169418931, -0.0599393845, 0.0329547822]]], [[[-0.052768141, -0.314567268, -0.186513826], [0.0532585084, 0.0245009959, 0.0529979467], [-0.331029862, -0.0602972806, 0.16982308]]], [[[-0.30826056, -0.136034891, 0.273737401], [-0.0011934936, -0.0771146118, -0.241945744], [-0.328254431, 0.110774368, -0.136318132]]], [[[0.328168541, -0.155555293, -0.155384392], [0.0929693281, -0.0594468117, -0.100888371], [-0.297718585, 0.164487213, 0.150235265]]], [[[0.234056145, 0.225965768, -0.0166739225], [-0.0740396678, -0.309546351, -0.289286137], [0.280160934, -0.0390281975, 0.309938341]]], [[[-0.137508512, -0.190572068, -0.238774583], [0.233296722, -0.273239464, 0.109690696], [0.134557009, -0.318382204, 0.140712619]]], [[[0.228845507, 0.298210949, -0.0441402197], [0.156154394, -0.0271853209, -0.258191198], [-0.107759565, -0.315619349, -0.104509875]]], [[[0.114422441, -0.185975239, -0.218470216], [0.137614936, -0.2726565, -0.023662895], [-0.0354144275, -0.0965298861, -0.192793533]]]]> : tensor<8x1x3x3xf32>) : !torch.vtensor<[8,1,3,3],f32>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty_list = torch.prim.ListConstruct : () -> !torch.list<int>
    %4 = torch.aten.convolution %arg0, %1, %0, %2, %3, %2, %false, %empty_list, %int1 {layer_name = "Conv_0"} : !torch.vtensor<[1,1,128,128],f32>, !torch.vtensor<[8,1,3,3],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,8,126,126],f32>
    return %4 : !torch.vtensor<[1,8,126,126],f32>
  }
}