// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file --verify-diagnostics

func.func @test_grid_sample_valid1(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 0 : i64, mode = 0 : i64, padding_mode = 0 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @test_grid_sample_valid2(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 0 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @test_grid_sample_valid3(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 0 : i64, mode = 0 : i64, padding_mode = 1 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @test_grid_sample_reflection(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 2 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @test_grid_sample_interpolate_nearest(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 1 : i64, mode = 1 : i64, padding_mode = 1 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @test_grid_sample_interpolate_cubic(%arg0: tensor<1x3x1152x1344xf32>, %arg1: tensor<1x1152x1344x2xf32>) -> tensor<*xf32> {
  %0 = "xten_nn.grid_sample"(%arg0, %arg1) {align_corners = 1 : i64, mode = 2 : i64, padding_mode = 1 : i64} : (tensor<1x3x1152x1344xf32>, tensor<1x1152x1344x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
