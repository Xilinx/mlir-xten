// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s
// REQUIRES: torch

func.func @reflect_pad_bf16(%arg0: tensor<1x32x122x122xbf16>) -> tensor<1x32x124x124xbf16> {
    %pad = "tosa.const"() <{value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>}> : () -> tensor<8xi64>
    %reflect_pad = xten_nn.reflect_pad %arg0, %pad {LayerName = "Pad_282", OutputName = "Pad_282"} : (tensor<1x32x122x122xbf16>, tensor<8xi64>) -> (tensor<1x32x124x124xbf16>)
    return %reflect_pad : tensor<1x32x124x124xbf16>
}
// CHECK-LABEL:  func.func @reflect_pad_bf16
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1x32x122x122xbf16>) -> tensor<1x32x124x124xbf16> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[PADS:.*]] = "tosa.const"() <{value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>}> : () -> tensor<8xi64>
// CHECK:    %[[FROM_ARG:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x32x122x122xbf16> -> !torch.vtensor<[1,32,122,122],bf16>
// CHECK:    %[[FROM_PADS:.*]] = torch_c.from_builtin_tensor %[[PADS]] : tensor<8xi64> -> !torch.vtensor<[8],si64>
// CHECK:    %[[OP:.*]] = torch.operator "onnx.Pad"(%[[FROM_ARG]], %[[FROM_PADS]]) {torch.onnx.mode = "reflect"} : (!torch.vtensor<[1,32,122,122],bf16>, !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,32,124,124],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,32,124,124],bf16> -> tensor<1x32x124x124xbf16>
// CHECK:    return %[[TO]] : tensor<1x32x124x124xbf16>

// -----

func.func @reflect_pad_bf16_3d(%arg0: tensor<32x122x122xbf16>) -> tensor<32x124x124xbf16> {
    %pad = "tosa.const"() <{value = dense<[0, 1, 1, 0, 1, 1]> : tensor<6xi64>}> : () -> tensor<6xi64>
    %reflect_pad = xten_nn.reflect_pad %arg0, %pad : (tensor<32x122x122xbf16>, tensor<6xi64>) -> (tensor<32x124x124xbf16>)
    return %reflect_pad : tensor<32x124x124xbf16>
}
// CHECK-LABEL:  func.func @reflect_pad_bf16
// CHECK-SAME:      (%[[ARG:.*]]: tensor<32x122x122xbf16>) -> tensor<32x124x124xbf16> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[PADS:.*]] = "tosa.const"() <{value = dense<[0, 1, 1, 0, 1, 1]> : tensor<6xi64>}> : () -> tensor<6xi64>
// CHECK:    %[[FROM_ARG:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<32x122x122xbf16> -> !torch.vtensor<[32,122,122],bf16>
// CHECK:    %[[FROM_PADS:.*]] = torch_c.from_builtin_tensor %[[PADS]] : tensor<6xi64> -> !torch.vtensor<[6],si64>
// CHECK:    %[[OP:.*]] = torch.operator "onnx.Pad"(%[[FROM_ARG]], %[[FROM_PADS]]) {torch.onnx.mode = "reflect"} : (!torch.vtensor<[32,122,122],bf16>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[32,124,124],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[32,124,124],bf16> -> tensor<32x124x124xbf16>
// CHECK:    return %[[TO]] : tensor<32x124x124xbf16>

// -----

func.func @reflect_pad_f32(%arg0: tensor<1x32x122x122xf32>) -> tensor<1x32x124x124xf32> {
    %pad = "tosa.const"() <{value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>}> : () -> tensor<8xi64>
    %reflect_pad = xten_nn.reflect_pad %arg0, %pad {LayerName = "Pad_282", OutputName = "Pad_282"} : (tensor<1x32x122x122xf32>, tensor<8xi64>) -> (tensor<1x32x124x124xf32>)
    return %reflect_pad : tensor<1x32x124x124xf32>
}
// CHECK-LABEL:  func.func @reflect_pad_f32
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1x32x122x122xf32>) -> tensor<1x32x124x124xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[PADS:.*]] = "tosa.const"() <{value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>}> : () -> tensor<8xi64>
// CHECK:    %[[FROM_ARG:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x32x122x122xf32> -> !torch.vtensor<[1,32,122,122],f32>
// CHECK:    %[[FROM_PADS:.*]] = torch_c.from_builtin_tensor %[[PADS]] : tensor<8xi64> -> !torch.vtensor<[8],si64>
// CHECK:    %[[OP:.*]] = torch.operator "onnx.Pad"(%[[FROM_ARG]], %[[FROM_PADS]]) {torch.onnx.mode = "reflect"} : (!torch.vtensor<[1,32,122,122],f32>, !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,32,124,124],f32>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,32,124,124],f32> -> tensor<1x32x124x124xf32>
// CHECK:    return %[[TO]] : tensor<1x32x124x124xf32>

// -----

func.func @reflect_pad_i32(%arg0: tensor<1x3x4x5xi32> , %arg1: tensor<8xi64>) -> tensor<1x3x6x7xi32> {
    %0 = xten_nn.reflect_pad %arg0, %arg1 : (tensor<1x3x4x5xi32>, tensor<8xi64>) -> tensor<1x3x6x7xi32>
    return %0 : tensor<1x3x6x7xi32>
}

// CHECK-LABEL: func @reflect_pad_i32
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<1x3x4x5xi32>, %[[ARG_1:.*]]: tensor<8xi64>) -> tensor<1x3x6x7xi32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[FROM_ARG_0:.*]] = torch_c.from_builtin_tensor %[[ARG_0]] : tensor<1x3x4x5xi32> -> !torch.vtensor<[1,3,4,5],si32>
// CHECK:    %[[FROM_ARG_1:.*]] = torch_c.from_builtin_tensor %[[ARG_1]] : tensor<8xi64> -> !torch.vtensor<[8],si64>
// CHECK:    %[[OP:.*]] = torch.operator "onnx.Pad"(%[[FROM_ARG_0]], %[[FROM_ARG_1]]) {torch.onnx.mode = "reflect"} : (!torch.vtensor<[1,3,4,5],si32>, !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,3,6,7],si32>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,3,6,7],si32> -> tensor<1x3x6x7xi32>
// CHECK:    return %[[TO]] : tensor<1x3x6x7xi32>

func.func @reflect_pad_i1(%arg0: tensor<1x3x4x5xi1> , %arg1: tensor<8xi64>) -> tensor<1x3x6x7xi1> {
    %0 = xten_nn.reflect_pad %arg0, %arg1 : (tensor<1x3x4x5xi1>, tensor<8xi64>) -> tensor<1x3x6x7xi1>
    return %0 : tensor<1x3x6x7xi1>
}

// CHECK-LABEL: func @reflect_pad_i1
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<1x3x4x5xi1>, %[[ARG_1:.*]]: tensor<8xi64>) -> tensor<1x3x6x7xi1> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:   %[[FROM_ARG_0:.*]] = torch_c.from_builtin_tensor %[[ARG_0]] : tensor<1x3x4x5xi1> -> !torch.vtensor<[1,3,4,5],i1>
// CHECK:   %[[FROM_ARG_1:.*]] = torch_c.from_builtin_tensor %[[ARG_1]] : tensor<8xi64> -> !torch.vtensor<[8],si64>
// CHECK:   %[[OP:.*]] = torch.operator "onnx.Pad"(%[[FROM_ARG_0]], %[[FROM_ARG_1]]) {torch.onnx.mode = "reflect"} : (!torch.vtensor<[1,3,4,5],i1>, !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,3,6,7],i1>
// CHECK:   %[[TO:.*]] = torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,3,6,7],i1> -> tensor<1x3x6x7xi1>
// CHECK:   return %[[TO]] : tensor<1x3x6x7xi1>