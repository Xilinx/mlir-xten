// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: not aten-opt --convert-xtennn-to-linalg %s 2>&1 | FileCheck %s

func.func @grid_sample_cubic_unsupported(%arg0: tensor<1x2x4x4xf32>, %arg1: tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32> {
  %0 = xten_nn.grid_sample %arg0, %arg1 {align_corners = 1 : i64, mode = 2 : i64, padding_mode = 0 : i64} : (tensor<1x2x4x4xf32>, tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}

// CHECK: failed to legalize operation 'xten_nn.grid_sample' that was explicitly marked illegal
// CHECK: mode = 2
