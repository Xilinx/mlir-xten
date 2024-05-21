// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @mish(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.mish %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[OP:.+]] = torch.aten.mish %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],f32> -> tensor<1x10xf32>
}


// -----

func.func @mish_bf16(%arg0: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.mish %arg0 : (tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[OP:.+]] = torch.aten.mish %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],bf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],bf16> -> tensor<1x10xbf16>
}
