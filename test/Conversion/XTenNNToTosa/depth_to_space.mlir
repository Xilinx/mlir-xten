// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s --xten-nn-to-tosa --split-input-file | FileCheck %s --check-prefix=PRESERVE
// RUN: aten-opt %s --xten-nn-to-tosa="enable-depth-to-space-decomposition=true" --split-input-file | FileCheck %s

func.func @depth_to_space_crd(%arg0: tensor<1x16x8x8xf32>) -> tensor<1x4x16x16xf32> {
  %0 = xten_nn.depth_to_space %arg0 {blocksize = 2 : i64, mode = 2 : i64} : (tensor<1x16x8x8xf32>) -> tensor<1x4x16x16xf32>
  return %0 : tensor<1x4x16x16xf32>
}

// CHECK-LABEL: func.func @depth_to_space_crd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x8x8xf32>) -> tensor<1x4x16x16xf32>
// CHECK-DAG: [[PERM:%.+]] = "tosa.const"() <{value = dense<[0, 1, 4, 2, 5, 3]> : tensor<6xi32>}> : () -> tensor<6xi32>
// CHECK-DAG: [[RESHAPED:%.+]] = tosa.reshape [[ARG0]] {new_shape = array<i64: 1, 4, 2, 2, 8, 8>} : (tensor<1x16x8x8xf32>) -> tensor<1x4x2x2x8x8xf32>
// CHECK: [[TRANSPOSED:%.+]] = tosa.transpose [[RESHAPED]], [[PERM]] : (tensor<1x4x2x2x8x8xf32>, tensor<6xi32>) -> tensor<1x4x8x2x8x2xf32>
// CHECK: [[RESULT:%.+]] = tosa.reshape [[TRANSPOSED]] {new_shape = array<i64: 1, 4, 16, 16>} : (tensor<1x4x8x2x8x2xf32>) -> tensor<1x4x16x16xf32>
// CHECK: return [[RESULT]] : tensor<1x4x16x16xf32>
// CHECK-NOT: xten_nn.depth_to_space

// PRESERVE-LABEL: func.func @depth_to_space_crd
// PRESERVE: xten_nn.depth_to_space

// -----

func.func @depth_to_space_dcr(%arg0: tensor<1x16x8x8xbf16>) -> tensor<1x4x16x16xbf16> {
  %0 = xten_nn.depth_to_space %arg0 {blocksize = 2 : i64, mode = 1 : i64} : (tensor<1x16x8x8xbf16>) -> tensor<1x4x16x16xbf16>
  return %0 : tensor<1x4x16x16xbf16>
}

// CHECK-LABEL: func.func @depth_to_space_dcr
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x8x8xbf16>) -> tensor<1x4x16x16xbf16>
// CHECK-DAG: [[PERM:%.+]] = "tosa.const"() <{value = dense<[0, 3, 4, 1, 5, 2]> : tensor<6xi32>}> : () -> tensor<6xi32>
// CHECK-DAG: [[RESHAPED:%.+]] = tosa.reshape [[ARG0]] {new_shape = array<i64: 1, 2, 2, 4, 8, 8>} : (tensor<1x16x8x8xbf16>) -> tensor<1x2x2x4x8x8xbf16>
// CHECK: [[TRANSPOSED:%.+]] = tosa.transpose [[RESHAPED]], [[PERM]] : (tensor<1x2x2x4x8x8xbf16>, tensor<6xi32>) -> tensor<1x4x8x2x8x2xbf16>
// CHECK: [[RESULT:%.+]] = tosa.reshape [[TRANSPOSED]] {new_shape = array<i64: 1, 4, 16, 16>} : (tensor<1x4x8x2x8x2xbf16>) -> tensor<1x4x16x16xbf16>
// CHECK: return [[RESULT]] : tensor<1x4x16x16xbf16>
// CHECK-NOT: xten_nn.depth_to_space

// PRESERVE-LABEL: func.func @depth_to_space_dcr
// PRESERVE: xten_nn.depth_to_space
