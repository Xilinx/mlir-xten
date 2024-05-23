// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @depth_to_space_CRD_bf16(%arg0: tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16> {
    %dts = xten_nn.depth_to_space %arg0 {blocksize = 2 : i64, mode = 2 : i64} : (tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16>
    return %dts : tensor<1x4x512x512xbf16>
}
// CHECK-LABEL:  func.func @depth_to_space_CRD_bf16
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16> {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x16x256x256xbf16> -> !torch.vtensor<[1,16,256,256],bf16>
// CHECK:    %[[DTS:.*]] = torch.operator "onnx.DepthToSpace"(%[[FROM]]) {torch.onnx.blocksize = 2 : i64, torch.onnx.mode = "CRD"} : (!torch.vtensor<[1,16,256,256],bf16>) -> !torch.vtensor<[1,4,512,512],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[DTS]] : !torch.vtensor<[1,4,512,512],bf16> -> tensor<1x4x512x512xbf16>
// CHECK:    return %[[TO]] : tensor<1x4x512x512xbf16>

// -----
func.func @depth_to_space_DCR_bf16(%arg0: tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16> {
    %dts = xten_nn.depth_to_space %arg0 {blocksize = 2 : i64, mode = 1 : i64} : (tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16>
    return %dts : tensor<1x4x512x512xbf16>
}
// CHECK-LABEL:  func.func @depth_to_space_DCR_bf16
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1x16x256x256xbf16>) -> tensor<1x4x512x512xbf16> {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x16x256x256xbf16> -> !torch.vtensor<[1,16,256,256],bf16>
// CHECK:    %[[DTS:.*]] = torch.operator "onnx.DepthToSpace"(%[[FROM]]) {torch.onnx.blocksize = 2 : i64, torch.onnx.mode = "DCR"} : (!torch.vtensor<[1,16,256,256],bf16>) -> !torch.vtensor<[1,4,512,512],bf16>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[DTS]] : !torch.vtensor<[1,4,512,512],bf16> -> tensor<1x4x512x512xbf16>

// -----
func.func @depth_to_space_DCR_f32(%arg0: tensor<1x16x256x256xf32>) -> tensor<1x4x512x512xf32> {
    %dts = xten_nn.depth_to_space %arg0 {blocksize = 2 : i64, mode = 1 : i64} : (tensor<1x16x256x256xf32>) -> tensor<1x4x512x512xf32>
    return %dts : tensor<1x4x512x512xf32>
}
// CHECK-LABEL:  func.func @depth_to_space_DCR_f32
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1x16x256x256xf32>) -> tensor<1x4x512x512xf32> {
// CHECK:    %[[FROM:.*]] = torch_c.from_builtin_tensor %[[ARG]] : tensor<1x16x256x256xf32> -> !torch.vtensor<[1,16,256,256],f32>
// CHECK:    %[[DTS:.*]] = torch.operator "onnx.DepthToSpace"(%[[FROM]]) {torch.onnx.blocksize = 2 : i64, torch.onnx.mode = "DCR"} : (!torch.vtensor<[1,16,256,256],f32>) -> !torch.vtensor<[1,4,512,512],f32>
// CHECK:    %[[TO:.*]] = torch_c.to_builtin_tensor %[[DTS]] : !torch.vtensor<[1,4,512,512],f32> -> tensor<1x4x512x512xf32>