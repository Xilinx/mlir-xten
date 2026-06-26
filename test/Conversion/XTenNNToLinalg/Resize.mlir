// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-linalg -split-input-file %s | FileCheck %s

func.func @resize_nearest_asymmetric_floor(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_asymmetric_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.divf [[VAR_4_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.floor [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.fptosi [[VAR_6_]] : f32 to i64
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_0_]], [[VAR_8_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_10_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[VAR_10_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i64
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.sitofp [[VAR_14_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_16_:%.+]] = arith.divf [[VAR_15_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_17_:%.+]] = math.floor [[VAR_16_]] : f32
// CHECK:             [[VAR_18_:%.+]] = arith.fptosi [[VAR_17_]] : f32 to i64
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[CST_0_2_]], [[VAR_19_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.cmpi sgt, [[VAR_21_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[CST_1_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_25_:%.+]] = arith.index_cast [[VAR_24_]] : index to i64
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.sitofp [[VAR_25_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_27_:%.+]] = arith.divf [[VAR_26_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_28_:%.+]] = math.floor [[VAR_27_]] : f32
// CHECK:             [[VAR_29_:%.+]] = arith.fptosi [[VAR_28_]] : f32 to i64
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_31_:%.+]] = arith.cmpi slt, [[VAR_30_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[CST_0_3_]], [[VAR_30_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.cmpi sgt, [[VAR_32_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.select [[VAR_33_]], [[CST_1_1_]], [[VAR_32_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_36_:%.+]] = arith.index_cast [[VAR_35_]] : index to i64
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.sitofp [[VAR_36_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_38_:%.+]] = arith.divf [[VAR_37_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_39_:%.+]] = math.floor [[VAR_38_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.fptosi [[VAR_39_]] : f32 to i64
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.index_cast [[VAR_40_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_42_:%.+]] = arith.cmpi slt, [[VAR_41_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_43_:%.+]] = arith.select [[VAR_42_]], [[CST_0_4_]], [[VAR_41_]] : index
// CHECK:             [[VAR_44_:%.+]] = arith.cmpi sgt, [[VAR_43_]], [[CST_1_2_]] : index
// CHECK:             [[VAR_45_:%.+]] = arith.select [[VAR_44_]], [[CST_1_2_]], [[VAR_43_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_12_]], [[VAR_23_]], [[VAR_34_]], [[VAR_45_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_nearest_half_pixel_round_prefer_ceil(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 0 : i64, nearest_mode = 1 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_half_pixel_round_prefer_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.divf [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.floor [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.fptosi [[VAR_8_]] : f32 to i64
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : i64 to index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpf oge, [[VAR_11_]], [[CST_5_dot_000000_1_]] : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_13_:%.+]] = arith.addi [[VAR_10_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_10_]] : index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_0_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_]], [[VAR_14_]] : index
// CHECK:             [[VAR_17_:%.+]] = arith.cmpi sgt, [[VAR_16_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_0_1_]], [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_19_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_20_:%.+]] = arith.index_cast [[VAR_19_]] : index to i64
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.sitofp [[VAR_20_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_2_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.divf [[VAR_22_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.subf [[VAR_23_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_25_:%.+]] = math.floor [[VAR_24_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.fptosi [[VAR_25_]] : f32 to i64
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.index_cast [[VAR_26_]] : i64 to index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.subf [[VAR_24_]], [[VAR_25_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_3_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.cmpf oge, [[VAR_28_]], [[CST_5_dot_000000_3_]] : f32
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_30_:%.+]] = arith.addi [[VAR_27_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_27_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_32_:%.+]] = arith.cmpi slt, [[VAR_31_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[CST_0_2_]], [[VAR_31_]] : index
// CHECK:             [[VAR_34_:%.+]] = arith.cmpi sgt, [[VAR_33_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_1_2_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_36_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_37_:%.+]] = arith.index_cast [[VAR_36_]] : index to i64
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.sitofp [[VAR_37_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_4_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.divf [[VAR_39_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_41_:%.+]] = arith.subf [[VAR_40_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.fptosi [[VAR_42_]] : f32 to i64
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.index_cast [[VAR_43_]] : i64 to index
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.subf [[VAR_41_]], [[VAR_42_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_5_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.cmpf oge, [[VAR_45_]], [[CST_5_dot_000000_5_]] : f32
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_47_:%.+]] = arith.addi [[VAR_44_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.select [[VAR_46_]], [[VAR_47_]], [[VAR_44_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_49_:%.+]] = arith.cmpi slt, [[VAR_48_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_50_:%.+]] = arith.select [[VAR_49_]], [[CST_0_3_]], [[VAR_48_]] : index
// CHECK:             [[VAR_51_:%.+]] = arith.cmpi sgt, [[VAR_50_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_1_4_]], [[VAR_50_]] : index
// CHECK-DAG:         [[VAR_53_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_54_:%.+]] = arith.index_cast [[VAR_53_]] : index to i64
// CHECK-DAG:         [[VAR_55_:%.+]] = arith.sitofp [[VAR_54_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_6_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_56_:%.+]] = arith.addf [[VAR_55_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_57_:%.+]] = arith.divf [[VAR_56_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_59_:%.+]] = math.floor [[VAR_58_]] : f32
// CHECK:             [[VAR_60_:%.+]] = arith.fptosi [[VAR_59_]] : f32 to i64
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.index_cast [[VAR_60_]] : i64 to index
// CHECK-DAG:         [[VAR_62_:%.+]] = arith.subf [[VAR_58_]], [[VAR_59_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_7_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.cmpf oge, [[VAR_62_]], [[CST_5_dot_000000_7_]] : f32
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_64_:%.+]] = arith.addi [[VAR_61_]], [[CST_1_5_]] : index
// CHECK-DAG:         [[VAR_65_:%.+]] = arith.select [[VAR_63_]], [[VAR_64_]], [[VAR_61_]] : index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_66_:%.+]] = arith.cmpi slt, [[VAR_65_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[CST_0_4_]], [[VAR_65_]] : index
// CHECK:             [[VAR_68_:%.+]] = arith.cmpi sgt, [[VAR_67_]], [[CST_1_6_]] : index
// CHECK:             [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[CST_1_6_]], [[VAR_67_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_52_]], [[VAR_69_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_nearest_half_pixel_round_prefer_floor(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 0 : i64, nearest_mode = 2 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_half_pixel_round_prefer_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.divf [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.floor [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.fptosi [[VAR_8_]] : f32 to i64
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : i64 to index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpf ogt, [[VAR_11_]], [[CST_5_dot_000000_1_]] : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_13_:%.+]] = arith.addi [[VAR_10_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_10_]] : index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_0_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_]], [[VAR_14_]] : index
// CHECK:             [[VAR_17_:%.+]] = arith.cmpi sgt, [[VAR_16_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_0_1_]], [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_19_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_20_:%.+]] = arith.index_cast [[VAR_19_]] : index to i64
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.sitofp [[VAR_20_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_2_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.divf [[VAR_22_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.subf [[VAR_23_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_25_:%.+]] = math.floor [[VAR_24_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.fptosi [[VAR_25_]] : f32 to i64
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.index_cast [[VAR_26_]] : i64 to index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.subf [[VAR_24_]], [[VAR_25_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_3_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.cmpf ogt, [[VAR_28_]], [[CST_5_dot_000000_3_]] : f32
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_30_:%.+]] = arith.addi [[VAR_27_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_27_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_32_:%.+]] = arith.cmpi slt, [[VAR_31_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[CST_0_2_]], [[VAR_31_]] : index
// CHECK:             [[VAR_34_:%.+]] = arith.cmpi sgt, [[VAR_33_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_1_2_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_36_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_37_:%.+]] = arith.index_cast [[VAR_36_]] : index to i64
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.sitofp [[VAR_37_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_4_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.divf [[VAR_39_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_41_:%.+]] = arith.subf [[VAR_40_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.fptosi [[VAR_42_]] : f32 to i64
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.index_cast [[VAR_43_]] : i64 to index
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.subf [[VAR_41_]], [[VAR_42_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_5_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.cmpf ogt, [[VAR_45_]], [[CST_5_dot_000000_5_]] : f32
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_47_:%.+]] = arith.addi [[VAR_44_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.select [[VAR_46_]], [[VAR_47_]], [[VAR_44_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_49_:%.+]] = arith.cmpi slt, [[VAR_48_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_50_:%.+]] = arith.select [[VAR_49_]], [[CST_0_3_]], [[VAR_48_]] : index
// CHECK:             [[VAR_51_:%.+]] = arith.cmpi sgt, [[VAR_50_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_1_4_]], [[VAR_50_]] : index
// CHECK-DAG:         [[VAR_53_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_54_:%.+]] = arith.index_cast [[VAR_53_]] : index to i64
// CHECK-DAG:         [[VAR_55_:%.+]] = arith.sitofp [[VAR_54_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_6_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_56_:%.+]] = arith.addf [[VAR_55_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_57_:%.+]] = arith.divf [[VAR_56_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_59_:%.+]] = math.floor [[VAR_58_]] : f32
// CHECK:             [[VAR_60_:%.+]] = arith.fptosi [[VAR_59_]] : f32 to i64
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.index_cast [[VAR_60_]] : i64 to index
// CHECK-DAG:         [[VAR_62_:%.+]] = arith.subf [[VAR_58_]], [[VAR_59_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_7_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.cmpf ogt, [[VAR_62_]], [[CST_5_dot_000000_7_]] : f32
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_64_:%.+]] = arith.addi [[VAR_61_]], [[CST_1_5_]] : index
// CHECK-DAG:         [[VAR_65_:%.+]] = arith.select [[VAR_63_]], [[VAR_64_]], [[VAR_61_]] : index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_66_:%.+]] = arith.cmpi slt, [[VAR_65_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[CST_0_4_]], [[VAR_65_]] : index
// CHECK:             [[VAR_68_:%.+]] = arith.cmpi sgt, [[VAR_67_]], [[CST_1_6_]] : index
// CHECK:             [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[CST_1_6_]], [[VAR_67_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_52_]], [[VAR_69_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_linear_half_pixel(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_linear_half_pixel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x1x4x4xf32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x1x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.divf [[VAR_5_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.floor [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.fptosi [[VAR_8_]] : f32 to i64
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : i64 to index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_14_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.cmpi sgt, [[VAR_15_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.select [[VAR_16_]], [[CST_0_1_]], [[VAR_15_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_18_:%.+]] = arith.cmpi slt, [[VAR_11_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[CST_0_2_]], [[VAR_11_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.cmpi sgt, [[VAR_19_]], [[CST_0_3_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[CST_0_3_]], [[VAR_19_]] : index
// CHECK-DAG:         [[VAR_22_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : index to i64
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.sitofp [[VAR_23_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_2_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_25_:%.+]] = arith.addf [[VAR_24_]], [[CST_5_dot_000000_1_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.divf [[VAR_25_]], [[CST_1_dot_000000_2_]] : f32
// CHECK:             [[VAR_27_:%.+]] = arith.subf [[VAR_26_]], [[CST_5_dot_000000_1_]] : f32
// CHECK:             [[VAR_28_:%.+]] = math.floor [[VAR_27_]] : f32
// CHECK:             [[VAR_29_:%.+]] = arith.fptosi [[VAR_28_]] : f32 to i64
// CHECK:             [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i64 to index
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.addi [[VAR_30_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.subf [[VAR_27_]], [[VAR_28_]] : f32
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_32_]] : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_34_:%.+]] = arith.cmpi slt, [[VAR_30_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_0_4_]], [[VAR_30_]] : index
// CHECK:             [[VAR_36_:%.+]] = arith.cmpi sgt, [[VAR_35_]], [[CST_0_5_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[CST_0_5_]], [[VAR_35_]] : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_38_:%.+]] = arith.cmpi slt, [[VAR_31_]], [[CST_0_6_]] : index
// CHECK:             [[VAR_39_:%.+]] = arith.select [[VAR_38_]], [[CST_0_6_]], [[VAR_31_]] : index
// CHECK:             [[VAR_40_:%.+]] = arith.cmpi sgt, [[VAR_39_]], [[CST_0_7_]] : index
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.select [[VAR_40_]], [[CST_0_7_]], [[VAR_39_]] : index
// CHECK-DAG:         [[VAR_42_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_43_:%.+]] = arith.index_cast [[VAR_42_]] : index to i64
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.sitofp [[VAR_43_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_2_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_45_:%.+]] = arith.addf [[VAR_44_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_46_:%.+]] = arith.divf [[VAR_45_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_47_:%.+]] = arith.subf [[VAR_46_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_48_:%.+]] = math.floor [[VAR_47_]] : f32
// CHECK:             [[VAR_49_:%.+]] = arith.fptosi [[VAR_48_]] : f32 to i64
// CHECK:             [[VAR_50_:%.+]] = arith.index_cast [[VAR_49_]] : i64 to index
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.addi [[VAR_50_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.subf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_54_:%.+]] = arith.cmpi slt, [[VAR_50_]], [[CST_0_8_]] : index
// CHECK:             [[VAR_55_:%.+]] = arith.select [[VAR_54_]], [[CST_0_8_]], [[VAR_50_]] : index
// CHECK:             [[VAR_56_:%.+]] = arith.cmpi sgt, [[VAR_55_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.select [[VAR_56_]], [[CST_1_1_]], [[VAR_55_]] : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_58_:%.+]] = arith.cmpi slt, [[VAR_51_]], [[CST_0_9_]] : index
// CHECK:             [[VAR_59_:%.+]] = arith.select [[VAR_58_]], [[CST_0_9_]], [[VAR_51_]] : index
// CHECK:             [[VAR_60_:%.+]] = arith.cmpi sgt, [[VAR_59_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.select [[VAR_60_]], [[CST_1_2_]], [[VAR_59_]] : index
// CHECK-DAG:         [[VAR_62_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_63_:%.+]] = arith.index_cast [[VAR_62_]] : index to i64
// CHECK-DAG:         [[VAR_64_:%.+]] = arith.sitofp [[VAR_63_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_3_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_65_:%.+]] = arith.addf [[VAR_64_]], [[CST_5_dot_000000_3_]] : f32
// CHECK:             [[VAR_66_:%.+]] = arith.divf [[VAR_65_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[CST_5_dot_000000_3_]] : f32
// CHECK:             [[VAR_68_:%.+]] = math.floor [[VAR_67_]] : f32
// CHECK:             [[VAR_69_:%.+]] = arith.fptosi [[VAR_68_]] : f32 to i64
// CHECK:             [[VAR_70_:%.+]] = arith.index_cast [[VAR_69_]] : i64 to index
// CHECK-DAG:         [[VAR_71_:%.+]] = arith.addi [[VAR_70_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_72_:%.+]] = arith.subf [[VAR_67_]], [[VAR_68_]] : f32
// CHECK-DAG:         [[VAR_73_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_74_:%.+]] = arith.cmpi slt, [[VAR_70_]], [[CST_0_10_]] : index
// CHECK:             [[VAR_75_:%.+]] = arith.select [[VAR_74_]], [[CST_0_10_]], [[VAR_70_]] : index
// CHECK:             [[VAR_76_:%.+]] = arith.cmpi sgt, [[VAR_75_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_77_:%.+]] = arith.select [[VAR_76_]], [[CST_1_3_]], [[VAR_75_]] : index
// CHECK-DAG:         [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_78_:%.+]] = arith.cmpi slt, [[VAR_71_]], [[CST_0_11_]] : index
// CHECK:             [[VAR_79_:%.+]] = arith.select [[VAR_78_]], [[CST_0_11_]], [[VAR_71_]] : index
// CHECK:             [[VAR_80_:%.+]] = arith.cmpi sgt, [[VAR_79_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_81_:%.+]] = arith.select [[VAR_80_]], [[CST_1_4_]], [[VAR_79_]] : index
// CHECK-DAG:         [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_3_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_82_:%.+]] = arith.mulf [[CST_1_dot_000000_3_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_83_:%.+]] = arith.mulf [[VAR_82_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_84_:%.+]] = arith.mulf [[VAR_83_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.mulf [[VAR_84_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_37_]], [[VAR_57_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_86_:%.+]] = arith.mulf [[VAR_extracted_]], [[VAR_85_]] : f32
// CHECK-DAG:         [[VAR_87_:%.+]] = arith.addf [[CST_0_dot_000000_]], [[VAR_86_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_4_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_88_:%.+]] = arith.mulf [[CST_1_dot_000000_4_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_89_:%.+]] = arith.mulf [[VAR_88_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_90_:%.+]] = arith.mulf [[VAR_89_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_91_:%.+]] = arith.mulf [[VAR_90_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_26_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_37_]], [[VAR_57_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_92_:%.+]] = arith.mulf [[VAR_extracted_26_]], [[VAR_91_]] : f32
// CHECK-DAG:         [[VAR_93_:%.+]] = arith.addf [[VAR_87_]], [[VAR_92_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_5_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_94_:%.+]] = arith.mulf [[CST_1_dot_000000_5_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_95_:%.+]] = arith.mulf [[VAR_94_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_96_:%.+]] = arith.mulf [[VAR_95_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_97_:%.+]] = arith.mulf [[VAR_96_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_28_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_41_]], [[VAR_57_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_98_:%.+]] = arith.mulf [[VAR_extracted_28_]], [[VAR_97_]] : f32
// CHECK-DAG:         [[VAR_99_:%.+]] = arith.addf [[VAR_93_]], [[VAR_98_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_6_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_100_:%.+]] = arith.mulf [[CST_1_dot_000000_6_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_101_:%.+]] = arith.mulf [[VAR_100_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_102_:%.+]] = arith.mulf [[VAR_101_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_103_:%.+]] = arith.mulf [[VAR_102_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_30_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_41_]], [[VAR_57_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_104_:%.+]] = arith.mulf [[VAR_extracted_30_]], [[VAR_103_]] : f32
// CHECK-DAG:         [[VAR_105_:%.+]] = arith.addf [[VAR_99_]], [[VAR_104_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_7_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_106_:%.+]] = arith.mulf [[CST_1_dot_000000_7_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_107_:%.+]] = arith.mulf [[VAR_106_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_108_:%.+]] = arith.mulf [[VAR_107_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_109_:%.+]] = arith.mulf [[VAR_108_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_32_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_37_]], [[VAR_61_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_110_:%.+]] = arith.mulf [[VAR_extracted_32_]], [[VAR_109_]] : f32
// CHECK-DAG:         [[VAR_111_:%.+]] = arith.addf [[VAR_105_]], [[VAR_110_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_8_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_112_:%.+]] = arith.mulf [[CST_1_dot_000000_8_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_113_:%.+]] = arith.mulf [[VAR_112_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_114_:%.+]] = arith.mulf [[VAR_113_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_115_:%.+]] = arith.mulf [[VAR_114_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_34_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_37_]], [[VAR_61_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_116_:%.+]] = arith.mulf [[VAR_extracted_34_]], [[VAR_115_]] : f32
// CHECK-DAG:         [[VAR_117_:%.+]] = arith.addf [[VAR_111_]], [[VAR_116_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_9_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_118_:%.+]] = arith.mulf [[CST_1_dot_000000_9_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_119_:%.+]] = arith.mulf [[VAR_118_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_120_:%.+]] = arith.mulf [[VAR_119_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_121_:%.+]] = arith.mulf [[VAR_120_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_36_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_41_]], [[VAR_61_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_122_:%.+]] = arith.mulf [[VAR_extracted_36_]], [[VAR_121_]] : f32
// CHECK-DAG:         [[VAR_123_:%.+]] = arith.addf [[VAR_117_]], [[VAR_122_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_10_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_124_:%.+]] = arith.mulf [[CST_1_dot_000000_10_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_125_:%.+]] = arith.mulf [[VAR_124_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_126_:%.+]] = arith.mulf [[VAR_125_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_127_:%.+]] = arith.mulf [[VAR_126_]], [[VAR_73_]] : f32
// CHECK-DAG:         [[VAR_extracted_38_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_41_]], [[VAR_61_]], [[VAR_77_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_128_:%.+]] = arith.mulf [[VAR_extracted_38_]], [[VAR_127_]] : f32
// CHECK-DAG:         [[VAR_129_:%.+]] = arith.addf [[VAR_123_]], [[VAR_128_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_11_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_130_:%.+]] = arith.mulf [[CST_1_dot_000000_11_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_131_:%.+]] = arith.mulf [[VAR_130_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_132_:%.+]] = arith.mulf [[VAR_131_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_133_:%.+]] = arith.mulf [[VAR_132_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_40_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_37_]], [[VAR_57_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_134_:%.+]] = arith.mulf [[VAR_extracted_40_]], [[VAR_133_]] : f32
// CHECK-DAG:         [[VAR_135_:%.+]] = arith.addf [[VAR_129_]], [[VAR_134_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_12_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_136_:%.+]] = arith.mulf [[CST_1_dot_000000_12_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_137_:%.+]] = arith.mulf [[VAR_136_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_138_:%.+]] = arith.mulf [[VAR_137_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_139_:%.+]] = arith.mulf [[VAR_138_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_42_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_37_]], [[VAR_57_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_140_:%.+]] = arith.mulf [[VAR_extracted_42_]], [[VAR_139_]] : f32
// CHECK-DAG:         [[VAR_141_:%.+]] = arith.addf [[VAR_135_]], [[VAR_140_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_13_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_142_:%.+]] = arith.mulf [[CST_1_dot_000000_13_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_143_:%.+]] = arith.mulf [[VAR_142_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_144_:%.+]] = arith.mulf [[VAR_143_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_145_:%.+]] = arith.mulf [[VAR_144_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_44_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_41_]], [[VAR_57_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_146_:%.+]] = arith.mulf [[VAR_extracted_44_]], [[VAR_145_]] : f32
// CHECK-DAG:         [[VAR_147_:%.+]] = arith.addf [[VAR_141_]], [[VAR_146_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_14_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_148_:%.+]] = arith.mulf [[CST_1_dot_000000_14_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_149_:%.+]] = arith.mulf [[VAR_148_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_150_:%.+]] = arith.mulf [[VAR_149_]], [[VAR_53_]] : f32
// CHECK-DAG:         [[VAR_151_:%.+]] = arith.mulf [[VAR_150_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_46_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_41_]], [[VAR_57_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_152_:%.+]] = arith.mulf [[VAR_extracted_46_]], [[VAR_151_]] : f32
// CHECK-DAG:         [[VAR_153_:%.+]] = arith.addf [[VAR_147_]], [[VAR_152_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_15_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_154_:%.+]] = arith.mulf [[CST_1_dot_000000_15_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_155_:%.+]] = arith.mulf [[VAR_154_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_156_:%.+]] = arith.mulf [[VAR_155_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_157_:%.+]] = arith.mulf [[VAR_156_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_48_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_37_]], [[VAR_61_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_158_:%.+]] = arith.mulf [[VAR_extracted_48_]], [[VAR_157_]] : f32
// CHECK-DAG:         [[VAR_159_:%.+]] = arith.addf [[VAR_153_]], [[VAR_158_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_16_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_160_:%.+]] = arith.mulf [[CST_1_dot_000000_16_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_161_:%.+]] = arith.mulf [[VAR_160_]], [[VAR_33_]] : f32
// CHECK:             [[VAR_162_:%.+]] = arith.mulf [[VAR_161_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_163_:%.+]] = arith.mulf [[VAR_162_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_50_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_37_]], [[VAR_61_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_164_:%.+]] = arith.mulf [[VAR_extracted_50_]], [[VAR_163_]] : f32
// CHECK-DAG:         [[VAR_165_:%.+]] = arith.addf [[VAR_159_]], [[VAR_164_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_17_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_166_:%.+]] = arith.mulf [[CST_1_dot_000000_17_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_167_:%.+]] = arith.mulf [[VAR_166_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_168_:%.+]] = arith.mulf [[VAR_167_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_169_:%.+]] = arith.mulf [[VAR_168_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_52_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_17_]], [[VAR_41_]], [[VAR_61_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_170_:%.+]] = arith.mulf [[VAR_extracted_52_]], [[VAR_169_]] : f32
// CHECK-DAG:         [[VAR_171_:%.+]] = arith.addf [[VAR_165_]], [[VAR_170_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_18_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_172_:%.+]] = arith.mulf [[CST_1_dot_000000_18_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_173_:%.+]] = arith.mulf [[VAR_172_]], [[VAR_32_]] : f32
// CHECK:             [[VAR_174_:%.+]] = arith.mulf [[VAR_173_]], [[VAR_52_]] : f32
// CHECK-DAG:         [[VAR_175_:%.+]] = arith.mulf [[VAR_174_]], [[VAR_72_]] : f32
// CHECK-DAG:         [[VAR_extracted_54_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_21_]], [[VAR_41_]], [[VAR_61_]], [[VAR_81_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_176_:%.+]] = arith.mulf [[VAR_extracted_54_]], [[VAR_175_]] : f32
// CHECK:             [[VAR_177_:%.+]] = arith.addf [[VAR_171_]], [[VAR_176_]] : f32
// CHECK:             linalg.yield [[VAR_177_]] : f32
// CHECK:           } -> tensor<1x1x4x4xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x1x4x4xf32>
// CHECK:         }

// -----

func.func @resize_linear_align_corners(%arg0: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 3 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16>
  return %0 : tensor<1x1x4x4xbf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_linear_align_corners
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x1x4x4xbf16>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x1x4x4xbf16>) {
// CHECK:           ^bb0([[OUT_:%.+]]: bf16):
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:             [[VAR_5_:%.+]] = math.floor [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.fptosi [[VAR_5_]] : f32 to i64
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i64 to index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_5_]] : f32
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_9_]] : f32
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[VAR_7_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_12_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_0_1_]], [[VAR_12_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_2_]], [[VAR_8_]] : index
// CHECK:             [[VAR_17_:%.+]] = arith.cmpi sgt, [[VAR_16_]], [[CST_0_3_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_0_3_]], [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_19_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_20_:%.+]] = arith.index_cast [[VAR_19_]] : index to i64
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.sitofp [[VAR_20_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_2_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:             [[VAR_22_:%.+]] = math.floor [[CST_0_dot_000000_1_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.fptosi [[VAR_22_]] : f32 to i64
// CHECK:             [[VAR_24_:%.+]] = arith.index_cast [[VAR_23_]] : i64 to index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_24_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.subf [[CST_0_dot_000000_1_]], [[VAR_22_]] : f32
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_26_]] : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_24_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[CST_0_4_]], [[VAR_24_]] : index
// CHECK:             [[VAR_30_:%.+]] = arith.cmpi sgt, [[VAR_29_]], [[CST_0_5_]] : index
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.select [[VAR_30_]], [[CST_0_5_]], [[VAR_29_]] : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_32_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_0_6_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[CST_0_6_]], [[VAR_25_]] : index
// CHECK:             [[VAR_34_:%.+]] = arith.cmpi sgt, [[VAR_33_]], [[CST_0_7_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_0_7_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_36_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_37_:%.+]] = arith.index_cast [[VAR_36_]] : index to i64
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.sitofp [[VAR_37_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_3_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:             [[VAR_39_:%.+]] = arith.mulf [[VAR_38_]], [[CST_1_dot_000000_3_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.divf [[VAR_39_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_41_:%.+]] = math.floor [[VAR_40_]] : f32
// CHECK:             [[VAR_42_:%.+]] = arith.fptosi [[VAR_41_]] : f32 to i64
// CHECK:             [[VAR_43_:%.+]] = arith.index_cast [[VAR_42_]] : i64 to index
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.addi [[VAR_43_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.subf [[VAR_40_]], [[VAR_41_]] : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_47_:%.+]] = arith.cmpi slt, [[VAR_43_]], [[CST_0_8_]] : index
// CHECK:             [[VAR_48_:%.+]] = arith.select [[VAR_47_]], [[CST_0_8_]], [[VAR_43_]] : index
// CHECK:             [[VAR_49_:%.+]] = arith.cmpi sgt, [[VAR_48_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_50_:%.+]] = arith.select [[VAR_49_]], [[CST_1_1_]], [[VAR_48_]] : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_51_:%.+]] = arith.cmpi slt, [[VAR_44_]], [[CST_0_9_]] : index
// CHECK:             [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_0_9_]], [[VAR_44_]] : index
// CHECK:             [[VAR_53_:%.+]] = arith.cmpi sgt, [[VAR_52_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[CST_1_2_]], [[VAR_52_]] : index
// CHECK-DAG:         [[VAR_55_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_56_:%.+]] = arith.index_cast [[VAR_55_]] : index to i64
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.sitofp [[VAR_56_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_4_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_3_dot_000000_1_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:             [[VAR_58_:%.+]] = arith.mulf [[VAR_57_]], [[CST_1_dot_000000_4_]] : f32
// CHECK:             [[VAR_59_:%.+]] = arith.divf [[VAR_58_]], [[CST_3_dot_000000_1_]] : f32
// CHECK:             [[VAR_60_:%.+]] = math.floor [[VAR_59_]] : f32
// CHECK:             [[VAR_61_:%.+]] = arith.fptosi [[VAR_60_]] : f32 to i64
// CHECK:             [[VAR_62_:%.+]] = arith.index_cast [[VAR_61_]] : i64 to index
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.addi [[VAR_62_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_64_:%.+]] = arith.subf [[VAR_59_]], [[VAR_60_]] : f32
// CHECK-DAG:         [[VAR_65_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_66_:%.+]] = arith.cmpi slt, [[VAR_62_]], [[CST_0_10_]] : index
// CHECK:             [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[CST_0_10_]], [[VAR_62_]] : index
// CHECK:             [[VAR_68_:%.+]] = arith.cmpi sgt, [[VAR_67_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[CST_1_3_]], [[VAR_67_]] : index
// CHECK-DAG:         [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_70_:%.+]] = arith.cmpi slt, [[VAR_63_]], [[CST_0_11_]] : index
// CHECK:             [[VAR_71_:%.+]] = arith.select [[VAR_70_]], [[CST_0_11_]], [[VAR_63_]] : index
// CHECK:             [[VAR_72_:%.+]] = arith.cmpi sgt, [[VAR_71_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_73_:%.+]] = arith.select [[VAR_72_]], [[CST_1_4_]], [[VAR_71_]] : index
// CHECK-DAG:         [[CST_0_dot_000000_2_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_5_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_74_:%.+]] = arith.mulf [[CST_1_dot_000000_5_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_75_:%.+]] = arith.mulf [[VAR_74_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_76_:%.+]] = arith.mulf [[VAR_75_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_77_:%.+]] = arith.mulf [[VAR_76_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_31_]], [[VAR_50_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_78_:%.+]] = arith.extf [[VAR_extracted_]] : bf16 to f32
// CHECK:             [[VAR_79_:%.+]] = arith.mulf [[VAR_78_]], [[VAR_77_]] : f32
// CHECK-DAG:         [[VAR_80_:%.+]] = arith.addf [[CST_0_dot_000000_2_]], [[VAR_79_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_6_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_81_:%.+]] = arith.mulf [[CST_1_dot_000000_6_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_82_:%.+]] = arith.mulf [[VAR_81_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_83_:%.+]] = arith.mulf [[VAR_82_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_84_:%.+]] = arith.mulf [[VAR_83_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_28_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_31_]], [[VAR_50_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_85_:%.+]] = arith.extf [[VAR_extracted_28_]] : bf16 to f32
// CHECK:             [[VAR_86_:%.+]] = arith.mulf [[VAR_85_]], [[VAR_84_]] : f32
// CHECK-DAG:         [[VAR_87_:%.+]] = arith.addf [[VAR_80_]], [[VAR_86_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_7_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_88_:%.+]] = arith.mulf [[CST_1_dot_000000_7_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_89_:%.+]] = arith.mulf [[VAR_88_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_90_:%.+]] = arith.mulf [[VAR_89_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_91_:%.+]] = arith.mulf [[VAR_90_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_30_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_35_]], [[VAR_50_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_92_:%.+]] = arith.extf [[VAR_extracted_30_]] : bf16 to f32
// CHECK:             [[VAR_93_:%.+]] = arith.mulf [[VAR_92_]], [[VAR_91_]] : f32
// CHECK-DAG:         [[VAR_94_:%.+]] = arith.addf [[VAR_87_]], [[VAR_93_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_8_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_95_:%.+]] = arith.mulf [[CST_1_dot_000000_8_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_96_:%.+]] = arith.mulf [[VAR_95_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_97_:%.+]] = arith.mulf [[VAR_96_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_98_:%.+]] = arith.mulf [[VAR_97_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_32_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_50_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_99_:%.+]] = arith.extf [[VAR_extracted_32_]] : bf16 to f32
// CHECK:             [[VAR_100_:%.+]] = arith.mulf [[VAR_99_]], [[VAR_98_]] : f32
// CHECK-DAG:         [[VAR_101_:%.+]] = arith.addf [[VAR_94_]], [[VAR_100_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_9_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_102_:%.+]] = arith.mulf [[CST_1_dot_000000_9_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_103_:%.+]] = arith.mulf [[VAR_102_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_104_:%.+]] = arith.mulf [[VAR_103_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_105_:%.+]] = arith.mulf [[VAR_104_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_34_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_31_]], [[VAR_54_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_106_:%.+]] = arith.extf [[VAR_extracted_34_]] : bf16 to f32
// CHECK:             [[VAR_107_:%.+]] = arith.mulf [[VAR_106_]], [[VAR_105_]] : f32
// CHECK-DAG:         [[VAR_108_:%.+]] = arith.addf [[VAR_101_]], [[VAR_107_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_10_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_109_:%.+]] = arith.mulf [[CST_1_dot_000000_10_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_110_:%.+]] = arith.mulf [[VAR_109_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_111_:%.+]] = arith.mulf [[VAR_110_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_112_:%.+]] = arith.mulf [[VAR_111_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_36_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_31_]], [[VAR_54_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_113_:%.+]] = arith.extf [[VAR_extracted_36_]] : bf16 to f32
// CHECK:             [[VAR_114_:%.+]] = arith.mulf [[VAR_113_]], [[VAR_112_]] : f32
// CHECK-DAG:         [[VAR_115_:%.+]] = arith.addf [[VAR_108_]], [[VAR_114_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_11_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_116_:%.+]] = arith.mulf [[CST_1_dot_000000_11_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_117_:%.+]] = arith.mulf [[VAR_116_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_118_:%.+]] = arith.mulf [[VAR_117_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_119_:%.+]] = arith.mulf [[VAR_118_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_38_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_35_]], [[VAR_54_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_120_:%.+]] = arith.extf [[VAR_extracted_38_]] : bf16 to f32
// CHECK:             [[VAR_121_:%.+]] = arith.mulf [[VAR_120_]], [[VAR_119_]] : f32
// CHECK-DAG:         [[VAR_122_:%.+]] = arith.addf [[VAR_115_]], [[VAR_121_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_12_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_123_:%.+]] = arith.mulf [[CST_1_dot_000000_12_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_124_:%.+]] = arith.mulf [[VAR_123_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_125_:%.+]] = arith.mulf [[VAR_124_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_126_:%.+]] = arith.mulf [[VAR_125_]], [[VAR_65_]] : f32
// CHECK-DAG:         [[VAR_extracted_40_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_54_]], [[VAR_69_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_127_:%.+]] = arith.extf [[VAR_extracted_40_]] : bf16 to f32
// CHECK:             [[VAR_128_:%.+]] = arith.mulf [[VAR_127_]], [[VAR_126_]] : f32
// CHECK-DAG:         [[VAR_129_:%.+]] = arith.addf [[VAR_122_]], [[VAR_128_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_13_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_130_:%.+]] = arith.mulf [[CST_1_dot_000000_13_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_131_:%.+]] = arith.mulf [[VAR_130_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_132_:%.+]] = arith.mulf [[VAR_131_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_133_:%.+]] = arith.mulf [[VAR_132_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_42_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_31_]], [[VAR_50_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_134_:%.+]] = arith.extf [[VAR_extracted_42_]] : bf16 to f32
// CHECK:             [[VAR_135_:%.+]] = arith.mulf [[VAR_134_]], [[VAR_133_]] : f32
// CHECK-DAG:         [[VAR_136_:%.+]] = arith.addf [[VAR_129_]], [[VAR_135_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_14_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_137_:%.+]] = arith.mulf [[CST_1_dot_000000_14_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_138_:%.+]] = arith.mulf [[VAR_137_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_139_:%.+]] = arith.mulf [[VAR_138_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_140_:%.+]] = arith.mulf [[VAR_139_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_44_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_31_]], [[VAR_50_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_141_:%.+]] = arith.extf [[VAR_extracted_44_]] : bf16 to f32
// CHECK:             [[VAR_142_:%.+]] = arith.mulf [[VAR_141_]], [[VAR_140_]] : f32
// CHECK-DAG:         [[VAR_143_:%.+]] = arith.addf [[VAR_136_]], [[VAR_142_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_15_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_144_:%.+]] = arith.mulf [[CST_1_dot_000000_15_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_145_:%.+]] = arith.mulf [[VAR_144_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_146_:%.+]] = arith.mulf [[VAR_145_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_147_:%.+]] = arith.mulf [[VAR_146_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_46_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_35_]], [[VAR_50_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_148_:%.+]] = arith.extf [[VAR_extracted_46_]] : bf16 to f32
// CHECK:             [[VAR_149_:%.+]] = arith.mulf [[VAR_148_]], [[VAR_147_]] : f32
// CHECK-DAG:         [[VAR_150_:%.+]] = arith.addf [[VAR_143_]], [[VAR_149_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_16_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_151_:%.+]] = arith.mulf [[CST_1_dot_000000_16_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_152_:%.+]] = arith.mulf [[VAR_151_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_153_:%.+]] = arith.mulf [[VAR_152_]], [[VAR_46_]] : f32
// CHECK-DAG:         [[VAR_154_:%.+]] = arith.mulf [[VAR_153_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_48_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_50_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_155_:%.+]] = arith.extf [[VAR_extracted_48_]] : bf16 to f32
// CHECK:             [[VAR_156_:%.+]] = arith.mulf [[VAR_155_]], [[VAR_154_]] : f32
// CHECK-DAG:         [[VAR_157_:%.+]] = arith.addf [[VAR_150_]], [[VAR_156_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_17_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_158_:%.+]] = arith.mulf [[CST_1_dot_000000_17_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_159_:%.+]] = arith.mulf [[VAR_158_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_160_:%.+]] = arith.mulf [[VAR_159_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_161_:%.+]] = arith.mulf [[VAR_160_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_50_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_31_]], [[VAR_54_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_162_:%.+]] = arith.extf [[VAR_extracted_50_]] : bf16 to f32
// CHECK:             [[VAR_163_:%.+]] = arith.mulf [[VAR_162_]], [[VAR_161_]] : f32
// CHECK-DAG:         [[VAR_164_:%.+]] = arith.addf [[VAR_157_]], [[VAR_163_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_18_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_165_:%.+]] = arith.mulf [[CST_1_dot_000000_18_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_166_:%.+]] = arith.mulf [[VAR_165_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_167_:%.+]] = arith.mulf [[VAR_166_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_168_:%.+]] = arith.mulf [[VAR_167_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_52_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_31_]], [[VAR_54_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_169_:%.+]] = arith.extf [[VAR_extracted_52_]] : bf16 to f32
// CHECK:             [[VAR_170_:%.+]] = arith.mulf [[VAR_169_]], [[VAR_168_]] : f32
// CHECK-DAG:         [[VAR_171_:%.+]] = arith.addf [[VAR_164_]], [[VAR_170_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_19_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_172_:%.+]] = arith.mulf [[CST_1_dot_000000_19_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_173_:%.+]] = arith.mulf [[VAR_172_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_174_:%.+]] = arith.mulf [[VAR_173_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_175_:%.+]] = arith.mulf [[VAR_174_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_54_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_35_]], [[VAR_54_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_176_:%.+]] = arith.extf [[VAR_extracted_54_]] : bf16 to f32
// CHECK:             [[VAR_177_:%.+]] = arith.mulf [[VAR_176_]], [[VAR_175_]] : f32
// CHECK-DAG:         [[VAR_178_:%.+]] = arith.addf [[VAR_171_]], [[VAR_177_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_20_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_179_:%.+]] = arith.mulf [[CST_1_dot_000000_20_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_180_:%.+]] = arith.mulf [[VAR_179_]], [[VAR_26_]] : f32
// CHECK:             [[VAR_181_:%.+]] = arith.mulf [[VAR_180_]], [[VAR_45_]] : f32
// CHECK-DAG:         [[VAR_182_:%.+]] = arith.mulf [[VAR_181_]], [[VAR_64_]] : f32
// CHECK-DAG:         [[VAR_extracted_56_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_18_]], [[VAR_35_]], [[VAR_54_]], [[VAR_73_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_183_:%.+]] = arith.extf [[VAR_extracted_56_]] : bf16 to f32
// CHECK:             [[VAR_184_:%.+]] = arith.mulf [[VAR_183_]], [[VAR_182_]] : f32
// CHECK:             [[VAR_185_:%.+]] = arith.addf [[VAR_178_]], [[VAR_184_]] : f32
// CHECK:             [[VAR_186_:%.+]] = arith.truncf [[VAR_185_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_186_]] : bf16
// CHECK:           } -> tensor<1x1x4x4xbf16>
// CHECK:           return [[VAR_1_]] : tensor<1x1x4x4xbf16>
// CHECK:         }

// -----

func.func @resize_nearest_5d(%arg0: tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0, 2.0>} : (tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16>
  return %0 : tensor<2x16x32x8x8xbf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL:  func.func @resize_nearest_5d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<2x16x32x8x8xbf16>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<2x16x32x8x8xbf16>) {
// CHECK:           ^bb0([[OUT_:%.+]]: bf16):
// CHECK:             [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.divf [[VAR_4_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.floor [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.fptosi [[VAR_6_]] : f32 to i64
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_0_]], [[VAR_8_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_10_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_1_]], [[VAR_10_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i64
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.sitofp [[VAR_14_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_16_:%.+]] = arith.divf [[VAR_15_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_17_:%.+]] = math.floor [[VAR_16_]] : f32
// CHECK:             [[VAR_18_:%.+]] = arith.fptosi [[VAR_17_]] : f32 to i64
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_]] : i64 to index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_15_:%.+]] = arith.constant 15 : index
// CHECK:             [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[CST_0_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[CST_0_1_]], [[VAR_19_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.cmpi sgt, [[VAR_21_]], [[CST_15_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[CST_15_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_25_:%.+]] = arith.index_cast [[VAR_24_]] : index to i64
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.sitofp [[VAR_25_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_27_:%.+]] = arith.divf [[VAR_26_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_28_:%.+]] = math.floor [[VAR_27_]] : f32
// CHECK:             [[VAR_29_:%.+]] = arith.fptosi [[VAR_28_]] : f32 to i64
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_15_1_:%.+]] = arith.constant 15 : index
// CHECK:             [[VAR_31_:%.+]] = arith.cmpi slt, [[VAR_30_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[CST_0_2_]], [[VAR_30_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.cmpi sgt, [[VAR_32_]], [[CST_15_1_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.select [[VAR_33_]], [[CST_15_1_]], [[VAR_32_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_36_:%.+]] = arith.index_cast [[VAR_35_]] : index to i64
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.sitofp [[VAR_36_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_38_:%.+]] = arith.divf [[VAR_37_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_39_:%.+]] = math.floor [[VAR_38_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.fptosi [[VAR_39_]] : f32 to i64
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.index_cast [[VAR_40_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:             [[VAR_42_:%.+]] = arith.cmpi slt, [[VAR_41_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_43_:%.+]] = arith.select [[VAR_42_]], [[CST_0_3_]], [[VAR_41_]] : index
// CHECK:             [[VAR_44_:%.+]] = arith.cmpi sgt, [[VAR_43_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.select [[VAR_44_]], [[CST_3_]], [[VAR_43_]] : index
// CHECK-DAG:         [[VAR_46_:%.+]] = linalg.index 4 : index
// CHECK:             [[VAR_47_:%.+]] = arith.index_cast [[VAR_46_]] : index to i64
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.sitofp [[VAR_47_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_2_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_49_:%.+]] = arith.divf [[VAR_48_]], [[CST_2_dot_000000_2_]] : f32
// CHECK:             [[VAR_50_:%.+]] = math.floor [[VAR_49_]] : f32
// CHECK:             [[VAR_51_:%.+]] = arith.fptosi [[VAR_50_]] : f32 to i64
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.index_cast [[VAR_51_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK:             [[VAR_53_:%.+]] = arith.cmpi slt, [[VAR_52_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[CST_0_4_]], [[VAR_52_]] : index
// CHECK:             [[VAR_55_:%.+]] = arith.cmpi sgt, [[VAR_54_]], [[CST_3_1_]] : index
// CHECK:             [[VAR_56_:%.+]] = arith.select [[VAR_55_]], [[CST_3_1_]], [[VAR_54_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_12_]], [[VAR_23_]], [[VAR_34_]], [[VAR_45_]], [[VAR_56_]]{{.}} : tensor<2x16x16x4x4xbf16>
// CHECK:             linalg.yield [[VAR_extracted_]] : bf16
// CHECK:           } -> tensor<2x16x32x8x8xbf16>
// CHECK:           return [[VAR_1_]] : tensor<2x16x32x8x8xbf16>
// CHECK:         }

// -----

func.func @resize_nearest_i8(%arg0: tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8>
  return %0 : tensor<1x2x4x4xi8>
}



// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_i8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8> {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1x2x4x4xi8>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_0_]] : tensor<1x2x4x4xi8>) {
// CHECK:           ^bb0([[OUT_:%.+]]: i8):
// CHECK:             [[VAR_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_5_:%.+]] = arith.divf [[VAR_4_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.floor [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.fptosi [[VAR_6_]] : f32 to i64
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_0_]], [[VAR_8_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_10_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[VAR_10_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[VAR_13_]] : index to i64
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.sitofp [[VAR_14_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_16_:%.+]] = arith.divf [[VAR_15_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_17_:%.+]] = math.floor [[VAR_16_]] : f32
// CHECK:             [[VAR_18_:%.+]] = arith.fptosi [[VAR_17_]] : f32 to i64
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[CST_0_2_]], [[VAR_19_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.cmpi sgt, [[VAR_21_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[CST_1_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_25_:%.+]] = arith.index_cast [[VAR_24_]] : index to i64
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.sitofp [[VAR_25_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_27_:%.+]] = arith.divf [[VAR_26_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_28_:%.+]] = math.floor [[VAR_27_]] : f32
// CHECK:             [[VAR_29_:%.+]] = arith.fptosi [[VAR_28_]] : f32 to i64
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_31_:%.+]] = arith.cmpi slt, [[VAR_30_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[CST_0_3_]], [[VAR_30_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.cmpi sgt, [[VAR_32_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.select [[VAR_33_]], [[CST_1_1_]], [[VAR_32_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_36_:%.+]] = arith.index_cast [[VAR_35_]] : index to i64
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.sitofp [[VAR_36_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_38_:%.+]] = arith.divf [[VAR_37_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_39_:%.+]] = math.floor [[VAR_38_]] : f32
// CHECK:             [[VAR_40_:%.+]] = arith.fptosi [[VAR_39_]] : f32 to i64
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.index_cast [[VAR_40_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_42_:%.+]] = arith.cmpi slt, [[VAR_41_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_43_:%.+]] = arith.select [[VAR_42_]], [[CST_0_4_]], [[VAR_41_]] : index
// CHECK:             [[VAR_44_:%.+]] = arith.cmpi sgt, [[VAR_43_]], [[CST_1_2_]] : index
// CHECK:             [[VAR_45_:%.+]] = arith.select [[VAR_44_]], [[CST_1_2_]], [[VAR_43_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_12_]], [[VAR_23_]], [[VAR_34_]], [[VAR_45_]]{{.}} : tensor<1x2x2x2xi8>
// CHECK:             linalg.yield [[VAR_extracted_]] : i8
// CHECK:           } -> tensor<1x2x4x4xi8>
// CHECK:           return [[VAR_1_]] : tensor<1x2x4x4xi8>
// CHECK:         }
