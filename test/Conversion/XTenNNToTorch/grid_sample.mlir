// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s
// REQUIRES: torch

func.func @gridsample_default_bf16(%arg0: tensor<1x32x576x384xbf16>, %arg1: tensor<1x64x16x2xbf16>) -> tensor<1x32x64x16xbf16> {
    %0 = xten_nn.grid_sample %arg0, %arg1 {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 0 : i64} : (tensor<1x32x576x384xbf16>, tensor<1x64x16x2xbf16>) -> tensor<1x32x64x16xbf16>
    return %0 : tensor<1x32x64x16xbf16>
}
// CHECK-LABEL:  func.func @gridsample_default_bf16
// CHECK-SAME:      (%[[ARG0:.*]]: tensor<1x32x576x384xbf16>, %[[ARG1:.*]]: tensor<1x64x16x2xbf16>) -> tensor<1x32x64x16xbf16> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[VAL_0:.*]] = torch_c.from_builtin_tensor %[[ARG0]] : tensor<1x32x576x384xbf16> -> !torch.vtensor<[1,32,576,384],bf16>
// CHECK:    %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[ARG1]] : tensor<1x64x16x2xbf16> -> !torch.vtensor<[1,64,16,2],bf16>
// CHECK:    %[[VAL_2:.*]] = torch.operator "onnx.GridSample"(%[[VAL_0]], %[[VAL_1]]) {torch.onnx.align_corners = 1 : si64, torch.onnx.mode = "bilinear", torch.onnx.padding_mode = "zeros"} : (!torch.vtensor<[1,32,576,384],bf16>, !torch.vtensor<[1,64,16,2],bf16>) -> !torch.vtensor<[1,32,64,16],bf16>
// CHECK:    %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,32,64,16],bf16> -> tensor<1x32x64x16xbf16>
// CHECK:    return %[[VAL_3]] : tensor<1x32x64x16xbf16>

// -----

func.func @gridsample_default_f32(%arg0: tensor<1x32x576x384xf32>, %arg1: tensor<1x64x16x2xf32>) -> tensor<1x32x64x16xf32> {
    %0 = xten_nn.grid_sample %arg0, %arg1 {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 0 : i64} : (tensor<1x32x576x384xf32>, tensor<1x64x16x2xf32>) -> tensor<1x32x64x16xf32>
    return %0 : tensor<1x32x64x16xf32>
}
// CHECK-LABEL:  func.func @gridsample_default_f32
// CHECK-SAME:      (%[[ARG0:.*]]: tensor<1x32x576x384xf32>, %[[ARG1:.*]]: tensor<1x64x16x2xf32>) -> tensor<1x32x64x16xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[VAL_0:.*]] = torch_c.from_builtin_tensor %[[ARG0]] : tensor<1x32x576x384xf32> -> !torch.vtensor<[1,32,576,384],f32>
// CHECK:    %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[ARG1]] : tensor<1x64x16x2xf32> -> !torch.vtensor<[1,64,16,2],f32>
// CHECK:    %[[VAL_2:.*]] = torch.operator "onnx.GridSample"(%[[VAL_0]], %[[VAL_1]]) {torch.onnx.align_corners = 1 : si64, torch.onnx.mode = "bilinear", torch.onnx.padding_mode = "zeros"} : (!torch.vtensor<[1,32,576,384],f32>, !torch.vtensor<[1,64,16,2],f32>) -> !torch.vtensor<[1,32,64,16],f32>
// CHECK:    %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,32,64,16],f32> -> tensor<1x32x64x16xf32>
