// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @resize_align_corners_bf16(%arg0: tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16> {
    %1 = xten_nn.resize %arg0 {coordinate_transformation_mode = 3 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00>} : (tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16>
    return %1 : tensor<1x256x32x32xbf16>
}
// CHECK-LABEL:  func.func @resize_align_corners_bf16
// CHECK-SAME:          (%[[ARG:.*]]: tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x256x16x16xbf16> -> !torch.vtensor<[1,256,16,16],bf16>
// CHECK:    %[[SCALES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK:    %[[NONE:.*]] = torch.constant.none
// CHECK:    %[[RESIZE:.*]] = torch.operator "onnx.Resize"(%[[FROM]], %[[NONE]], %[[SCALES]]) {torch.onnx.coordinate_transformation_mode = "align_corners", torch.onnx.mode = "linear"} : (!torch.vtensor<[1,256,16,16],bf16>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,256,32,32],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[RESIZE]] : !torch.vtensor<[1,256,32,32],bf16> -> tensor<1x256x32x32xbf16>
// CHECK:    return %[[TO]] : tensor<1x256x32x32xbf16>

// -----
func.func @resize_pt_half_pixel_bf16(%arg0: tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16> {
    %1 = xten_nn.resize %arg0 {coordinate_transformation_mode = 1 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00>} : (tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16>
    return %1 : tensor<1x256x32x32xbf16>
}
// CHECK-LABEL:  func.func @resize_pt_half_pixel_bf16
// CHECK-SAME:          (%[[ARG:.*]]: tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x256x16x16xbf16> -> !torch.vtensor<[1,256,16,16],bf16>
// CHECK:    %[[SCALES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK:    %[[NONE:.*]] = torch.constant.none
// CHECK:    %[[RESIZE:.*]] = torch.operator "onnx.Resize"(%[[FROM]], %[[NONE]], %[[SCALES]]) {torch.onnx.coordinate_transformation_mode = "pytorch_half_pixel", torch.onnx.mode = "linear"} : (!torch.vtensor<[1,256,16,16],bf16>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,256,32,32],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[RESIZE]] : !torch.vtensor<[1,256,32,32],bf16> -> tensor<1x256x32x32xbf16>
// CHECK:    return %[[TO]] : tensor<1x256x32x32xbf16>

// -----
func.func @resize_half_pixel_f32(%arg0: tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32> {
    %1 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32>
    return %1 : tensor<1x256x32x32xf32>
}
// CHECK-LABEL:  func.func @resize_half_pixel_f32
// CHECK-SAME:          (%[[ARG:.*]]: tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x256x16x16xf32> -> !torch.vtensor<[1,256,16,16],f32>
// CHECK:    %[[SCALES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK:    %[[NONE:.*]] = torch.constant.none
// CHECK:    %[[RESIZE:.*]] = torch.operator "onnx.Resize"(%[[FROM]], %[[NONE]], %[[SCALES]]) {torch.onnx.coordinate_transformation_mode = "half_pixel", torch.onnx.mode = "linear"} : (!torch.vtensor<[1,256,16,16],f32>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,256,32,32],f32>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[RESIZE]] : !torch.vtensor<[1,256,32,32],f32> -> tensor<1x256x32x32xf32>
// CHECK:    return %[[TO]] : tensor<1x256x32x32xf32>

// -----
// Mode 'nearest' is to be lowered to tosa so we reject it here.
func.func @resize_nearest(%arg0: tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32> {
    %1 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32>
    return %1 : tensor<1x256x32x32xf32>
}
// CHECK-LABEL:  func.func @resize_nearest
// CHECK-SAME:          (%[[ARG:.*]]: tensor<1x256x16x16xf32>) -> tensor<1x256x32x32xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x256x16x16xf32> -> !torch.vtensor<[1,256,16,16],f32>
// CHECK:    %[[SCALES:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
// CHECK:    %[[NONE:.*]] = torch.constant.none
// CHECK:    %[[RESIZE:.*]] = torch.operator "onnx.Resize"(%[[FROM]], %[[NONE]], %[[SCALES]]) {torch.onnx.coordinate_transformation_mode = "half_pixel", torch.onnx.mode = "nearest"} : (!torch.vtensor<[1,256,16,16],f32>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,256,32,32],f32>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[RESIZE]] : !torch.vtensor<[1,256,32,32],f32> -> tensor<1x256x32x32xf32>
// CHECK:    return %[[TO]] : tensor<1x256x32x32xf32>

