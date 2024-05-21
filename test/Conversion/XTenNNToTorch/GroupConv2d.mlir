// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch -split-input-file %s | FileCheck %s

func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1: tensor<12x16x45x45xf32>) -> tensor<5x12x17x17xf32> {
    %0 = arith.constant dense<0.000000e+00> : tensor<12xf32>
    %1 = xten_nn.group_conv2d %arg0, %arg1, %0 {dilation = array<i64: 13, 13>, group = 4 : i64, pad = [array<i64: 2, 2>, array<i64: 0, 0>], stride = array<i64: 1, 1>} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) -> tensor<5x12x17x17xf32>
    return %1 : tensor<5x12x17x17xf32>
//CHECK-LABEL: @test_onnx_conv2d_group
//      CHECK: %[[VAL_0:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<5x64x256x256xf32> -> !torch.vtensor<[5,64,256,256],f32>
// CHECK-NEXT: %[[VAL_1:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<12x16x45x45xf32> -> !torch.vtensor<[12,16,45,45],f32>
// CHECK-NEXT: %[[VAL_2:.+]] = torch_c.from_builtin_tensor %cst : tensor<12xf32> -> !torch.vtensor<[12],f32>
//  CHECK-DAG: %int1 = torch.constant.int 1
//  CHECK-DAG: %int1_0 = torch.constant.int 1
//  CHECK-DAG: %[[VAL_3:.+]] = torch.prim.ListConstruct %int1, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG: %int13 = torch.constant.int 13
//  CHECK-DAG: %int13_1 = torch.constant.int 13
//  CHECK-DAG: %[[VAL_4:.+]] = torch.prim.ListConstruct %int13, %int13_1 : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG: %int2 = torch.constant.int 2
//  CHECK-DAG: %int0 = torch.constant.int 0
//  CHECK-DAG: %[[VAL_5:.+]] = torch.prim.ListConstruct %int2, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG: %int4 = torch.constant.int 4
// CHECK-NEXT: %[[VAL_6:.+]] = torch.aten.conv2d %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_5]], %[[VAL_4]], %int4 : !torch.vtensor<[5,64,256,256],f32>, !torch.vtensor<[12,16,45,45],f32>, !torch.vtensor<[12],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[5,12,17,17],f32>
// CHECK-NEXT: %[[VAL_7:.+]] = torch_c.to_builtin_tensor %[[VAL_6]] : !torch.vtensor<[5,12,17,17],f32> -> tensor<5x12x17x17xf32>
// CHECK-NEXT: return %[[VAL_7]] : tensor<5x12x17x17xf32>
}

func.func @test_onnx_conv2d_group_asymmetric_pads(%arg0: tensor<5x64x255x253xf32>, %arg1: tensor<12x16x45x45xf32>) -> tensor<5x12x17x17xf32> {
    %0 = arith.constant dense<0.000000e+00> : tensor<12xf32>
    %1 = xten_nn.group_conv2d %arg0, %arg1, %0 {dilation = array<i64: 13, 13>, group = 4 : i64, pad = [array<i64: 1, 0>, array<i64: 0, 3>], stride = array<i64: 1, 1>} : (tensor<5x64x255x253xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) -> tensor<5x12x17x17xf32>
    return %1 : tensor<5x12x17x17xf32>
}

//CHECK-LABEL: @test_onnx_conv2d_group_asymmetric_pads(
//      CHECK:   %[[VAL_3:.*]] = torch_c.from_builtin_tensor %arg0 : tensor<5x64x255x253xf32> -> !torch.vtensor<[5,64,255,253],f32>
// CHECK-NEXT:   %[[VAL_4:.*]] = torch_c.from_builtin_tensor %arg1 : tensor<12x16x45x45xf32> -> !torch.vtensor<[12,16,45,45],f32>
// CHECK-NEXT:   %[[VAL_5:.*]] = torch_c.from_builtin_tensor %cst : tensor<12xf32> -> !torch.vtensor<[12],f32>
//  CHECK-DAG:   %[[VAL_6:.*]] = torch.constant.int 0
//  CHECK-DAG:   %[[VAL_7:.*]] = torch.constant.int 1
//  CHECK-DAG:   %[[VAL_8:.*]] = torch.constant.int 0
//  CHECK-DAG:   %[[VAL_9:.*]] = torch.constant.int 0
//  CHECK-DAG:   %[[VAL_10:.*]] = torch.constant.int 3
//  CHECK-DAG:   %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:   %[[VAL_12:.*]] = torch.aten.constant_pad_nd %[[VAL_3]], %[[VAL_11]], %[[VAL_6]] : !torch.vtensor<[5,64,255,253],f32>, !torch.list<int>, !torch.int -> !torch.vtensor<[5,64,256,256],f32>
//  CHECK-DAG:   %[[VAL_13:.*]] = torch.constant.int 0
//  CHECK-DAG:   %[[VAL_14:.*]] = torch.constant.int 0
//  CHECK-DAG:   %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG:   %[[VAL_16:.*]] = torch.constant.int 1
//  CHECK-DAG:   %[[VAL_17:.*]] = torch.constant.int 1
//  CHECK-DAG:   %[[VAL_18:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG:   %[[VAL_19:.*]] = torch.constant.int 13
//  CHECK-DAG:   %[[VAL_20:.*]] = torch.constant.int 13
//  CHECK-DAG:   %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_19]], %[[VAL_20]] : (!torch.int, !torch.int) -> !torch.list<int>
//  CHECK-DAG:   %[[VAL_22:.*]] = torch.constant.int 4
// CHECK-NEXT:   %[[VAL_23:.*]] = torch.aten.conv2d %[[VAL_12]], %[[VAL_4]], %[[VAL_5]], %[[VAL_18]], %[[VAL_15]], %[[VAL_21]], %[[VAL_22]] : !torch.vtensor<[5,64,256,256],f32>, !torch.vtensor<[12,16,45,45],f32>, !torch.vtensor<[12],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[5,12,17,17],f32>
// CHECK-NEXT:   %[[VAL_24:.*]] = torch_c.to_builtin_tensor %[[VAL_23]] : !torch.vtensor<[5,12,17,17],f32> -> tensor<5x12x17x17xf32>
// CHECK-NEXT:   return %[[VAL_24]] : tensor<5x12x17x17xf32>
