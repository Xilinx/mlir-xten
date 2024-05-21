// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @atan2_bf16(%arg0: tensor<1x10xbf16>, %arg1: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.atan2 %arg0, %arg1 : (tensor<1x10xbf16>, tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[FROM_BUILTIN2:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[OP:.+]] = torch.aten.atan2 %[[FROM_BUILTIN]], %[[FROM_BUILTIN2]] : !torch.vtensor<[1,10],bf16>, !torch.vtensor<[1,10],bf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],bf16> -> tensor<1x10xbf16>
}

// -----

func.func @atan2_f32(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.atan2 %arg0, %arg1 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[FROM_BUILTIN2:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[OP:.+]] = torch.aten.atan2 %[[FROM_BUILTIN]], %[[FROM_BUILTIN2]] : !torch.vtensor<[1,10],f32>, !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],f32> -> tensor<1x10xf32>
}

// -----

func.func @cos_bf16(%arg0: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.cos %arg0 : (tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[OP:.+]] = torch.aten.cos %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],bf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],bf16> -> tensor<1x10xbf16>
}

// -----

func.func @cos_f32(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.cos %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[OP:.+]] = torch.aten.cos %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],f32> -> tensor<1x10xf32>
}

// -----

func.func @sin_bf16(%arg0: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.sin %arg0 : (tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[OP:.+]] = torch.aten.sin %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],bf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],bf16> -> tensor<1x10xbf16>
}

// -----

func.func @sin_f32(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.sin %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[OP:.+]] = torch.aten.sin %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],f32> -> tensor<1x10xf32>
}