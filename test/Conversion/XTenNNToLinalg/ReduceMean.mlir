// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-linalg -split-input-file %s | FileCheck %s

func.func @reduce_mean_one_axis_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 2>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32>
  return %0 : tensor<4x512x1x8xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:  func.func @reduce_mean_one_axis_keep_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4x512x8xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<4x512x8xf32>) -> tensor<4x512x8xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[PARAM_0_]] : tensor<4x512x256x8xf32>) outs([[VAR_1_]] : tensor<4x512x8xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_8_:%.+]] = arith.addf [[IN_]], [[OUT_]] : f32
// CHECK:             linalg.yield [[VAR_8_]] : f32
// CHECK:           } -> tensor<4x512x8xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK:           [[VAR_3_:%.+]] = arith.muli [[CST_1_]], [[CST_256_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[VAR_6_:%.+]] = tensor.empty() : tensor<4x512x8xf32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_2_]] : tensor<4x512x8xf32>) outs([[VAR_6_]] : tensor<4x512x8xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_8_1_:%.+]] = arith.divf [[IN_]], [[VAR_5_]] : f32
// CHECK:             linalg.yield [[VAR_8_1_]] : f32
// CHECK:           } -> tensor<4x512x8xf32>
// CHECK:           [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_7_]] {{.}}[0], [1], [2, 3]{{.}} output_shape [4, 512, 1, 8] : tensor<4x512x8xf32> into tensor<4x512x1x8xf32>
// CHECK:           return [[VAR_expanded_]] : tensor<4x512x1x8xf32>
// CHECK:         }

// -----

func.func @reduce_mean_three_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @reduce_mean_three_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins([[PARAM_0_]] : tensor<4x512x256x8xf32>) outs([[VAR_1_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[IN_]], [[OUT_]] : f32
// CHECK:             linalg.yield [[VAR_10_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[CST_1_]], [[CST_512_]] : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_256_]] : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK:           [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[CST_8_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.sitofp [[VAR_6_]] : i64 to f32
// CHECK-DAG:       [[VAR_8_:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins([[VAR_2_]] : tensor<4xf32>) outs([[VAR_8_]] : tensor<4xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_10_1_:%.+]] = arith.divf [[IN_]], [[VAR_7_]] : f32
// CHECK:             linalg.yield [[VAR_10_1_]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return [[VAR_9_]] : tensor<4xf32>
// CHECK:         }

// -----

func.func @reduce_mean_all_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<f32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-LABEL:  func.func @reduce_mean_all_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} ins([[PARAM_0_]] : tensor<4x512x256x8xf32>) outs([[VAR_1_]] : tensor<f32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[IN_]], [[OUT_]] : f32
// CHECK:             linalg.yield [[VAR_10_]] : f32
// CHECK:           } -> tensor<f32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[CST_1_]], [[CST_4_]] : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_512_]] : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[CST_256_]] : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_8_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : index to i64
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.sitofp [[VAR_7_]] : i64 to f32
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = arith.divf [[VAR_extracted_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_9_]] : tensor<f32>
// CHECK:           return [[VAR_from_elements_]] : tensor<f32>
// CHECK:         }

// -----

func.func @reduce_mean_all_axes_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-LABEL:  func.func @reduce_mean_all_axes_keep_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} ins([[PARAM_0_]] : tensor<4x512x256x8xf32>) outs([[VAR_1_]] : tensor<f32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_12_:%.+]] = arith.addf [[IN_]], [[OUT_]] : f32
// CHECK:             linalg.yield [[VAR_12_]] : f32
// CHECK:           } -> tensor<f32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[CST_1_]], [[CST_4_]] : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_512_]] : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[CST_256_]] : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_8_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : index to i64
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.sitofp [[VAR_7_]] : i64 to f32
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = arith.divf [[VAR_extracted_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_9_]] : tensor<f32>
// CHECK-DAG:       [[VAR_extracted_0_:%.+]] = tensor.extract [[VAR_from_elements_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK:           [[VAR_11_:%.+]] = linalg.fill ins([[VAR_extracted_0_]] : f32) outs([[VAR_10_]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           return [[VAR_11_]] : tensor<1x1x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_mean_noop_with_empty_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32>
  return %0 : tensor<4x512x256x8xf32>
}

// CHECK-LABEL:  func.func @reduce_mean_noop_with_empty_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
// CHECK:           return [[PARAM_0_]] : tensor<4x512x256x8xf32>
// CHECK:         }

// -----

func.func @reduce_mean_negative_axis(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: -2>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32>
  return %0 : tensor<4x512x1x8xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:  func.func @reduce_mean_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4x512x8xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<4x512x8xf32>) -> tensor<4x512x8xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[PARAM_0_]] : tensor<4x512x256x8xf32>) outs([[VAR_1_]] : tensor<4x512x8xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_8_:%.+]] = arith.addf [[IN_]], [[OUT_]] : f32
// CHECK:             linalg.yield [[VAR_8_]] : f32
// CHECK:           } -> tensor<4x512x8xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK:           [[VAR_3_:%.+]] = arith.muli [[CST_1_]], [[CST_256_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[VAR_6_:%.+]] = tensor.empty() : tensor<4x512x8xf32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_2_]] : tensor<4x512x8xf32>) outs([[VAR_6_]] : tensor<4x512x8xf32>) {
// CHECK:           ^bb0([[IN_:%.+]]: f32, [[OUT_:%.+]]: f32):
// CHECK:             [[VAR_8_1_:%.+]] = arith.divf [[IN_]], [[VAR_5_]] : f32
// CHECK:             linalg.yield [[VAR_8_1_]] : f32
// CHECK:           } -> tensor<4x512x8xf32>
// CHECK:           [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_7_]] {{.}}[0], [1], [2, 3]{{.}} output_shape [4, 512, 1, 8] : tensor<4x512x8xf32> into tensor<4x512x1x8xf32>
// CHECK:           return [[VAR_expanded_]] : tensor<4x512x1x8xf32>
// CHECK:         }
