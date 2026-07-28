// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-linalg -split-input-file %s | FileCheck %s

func.func @resize_nearest_asymmetric_floor(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_asymmetric_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_v003_:%.+]] = arith.index_cast [[VAR_v002_]] : index to i64
// CHECK-DAG:         [[VAR_v004_:%.+]] = arith.sitofp [[VAR_v003_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v005_:%.+]] = arith.divf [[VAR_v004_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_v006_:%.+]] = math.floor [[VAR_v005_]] : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.fptosi [[VAR_v006_]] : f32 to i64
// CHECK-DAG:         [[VAR_v008_:%.+]] = arith.index_cast [[VAR_v007_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_v009_:%.+]] = arith.cmpi slt, [[VAR_v008_]], [[CST_0_]] : index
// CHECK:             [[VAR_v010_:%.+]] = arith.select [[VAR_v009_]], [[CST_0_]], [[VAR_v008_]] : index
// CHECK:             [[VAR_v011_:%.+]] = arith.cmpi sgt, [[VAR_v010_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.select [[VAR_v011_]], [[CST_0_1_]], [[VAR_v010_]] : index
// CHECK-DAG:         [[VAR_v013_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_v014_:%.+]] = arith.index_cast [[VAR_v013_]] : index to i64
// CHECK-DAG:         [[VAR_v015_:%.+]] = arith.sitofp [[VAR_v014_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v016_:%.+]] = arith.divf [[VAR_v015_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v017_:%.+]] = math.floor [[VAR_v016_]] : f32
// CHECK:             [[VAR_v018_:%.+]] = arith.fptosi [[VAR_v017_]] : f32 to i64
// CHECK-DAG:         [[VAR_v019_:%.+]] = arith.index_cast [[VAR_v018_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.cmpi slt, [[VAR_v019_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v021_:%.+]] = arith.select [[VAR_v020_]], [[CST_0_2_]], [[VAR_v019_]] : index
// CHECK:             [[VAR_v022_:%.+]] = arith.cmpi sgt, [[VAR_v021_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v023_:%.+]] = arith.select [[VAR_v022_]], [[CST_1_]], [[VAR_v021_]] : index
// CHECK-DAG:         [[VAR_v024_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v025_:%.+]] = arith.index_cast [[VAR_v024_]] : index to i64
// CHECK-DAG:         [[VAR_v026_:%.+]] = arith.sitofp [[VAR_v025_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v027_:%.+]] = arith.divf [[VAR_v026_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v028_:%.+]] = math.floor [[VAR_v027_]] : f32
// CHECK:             [[VAR_v029_:%.+]] = arith.fptosi [[VAR_v028_]] : f32 to i64
// CHECK-DAG:         [[VAR_v030_:%.+]] = arith.index_cast [[VAR_v029_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v031_:%.+]] = arith.cmpi slt, [[VAR_v030_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v032_:%.+]] = arith.select [[VAR_v031_]], [[CST_0_3_]], [[VAR_v030_]] : index
// CHECK:             [[VAR_v033_:%.+]] = arith.cmpi sgt, [[VAR_v032_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v034_:%.+]] = arith.select [[VAR_v033_]], [[CST_1_1_]], [[VAR_v032_]] : index
// CHECK-DAG:         [[VAR_v035_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v036_:%.+]] = arith.index_cast [[VAR_v035_]] : index to i64
// CHECK-DAG:         [[VAR_v037_:%.+]] = arith.sitofp [[VAR_v036_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v038_:%.+]] = arith.divf [[VAR_v037_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v039_:%.+]] = math.floor [[VAR_v038_]] : f32
// CHECK:             [[VAR_v040_:%.+]] = arith.fptosi [[VAR_v039_]] : f32 to i64
// CHECK-DAG:         [[VAR_v041_:%.+]] = arith.index_cast [[VAR_v040_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v042_:%.+]] = arith.cmpi slt, [[VAR_v041_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_v043_:%.+]] = arith.select [[VAR_v042_]], [[CST_0_4_]], [[VAR_v041_]] : index
// CHECK:             [[VAR_v044_:%.+]] = arith.cmpi sgt, [[VAR_v043_]], [[CST_1_2_]] : index
// CHECK:             [[VAR_v045_:%.+]] = arith.select [[VAR_v044_]], [[CST_1_2_]], [[VAR_v043_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v012_]], [[VAR_v023_]], [[VAR_v034_]], [[VAR_v045_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_v001_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_nearest_half_pixel_round_prefer_ceil(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 0 : i64, nearest_mode = 1 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_half_pixel_round_prefer_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_v003_:%.+]] = arith.index_cast [[VAR_v002_]] : index to i64
// CHECK-DAG:         [[VAR_v004_:%.+]] = arith.sitofp [[VAR_v003_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v005_:%.+]] = arith.addf [[VAR_v004_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v006_:%.+]] = arith.divf [[VAR_v005_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.subf [[VAR_v006_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v008_:%.+]] = math.floor [[VAR_v007_]] : f32
// CHECK:             [[VAR_v009_:%.+]] = arith.fptosi [[VAR_v008_]] : f32 to i64
// CHECK-DAG:         [[VAR_v010_:%.+]] = arith.index_cast [[VAR_v009_]] : i64 to index
// CHECK-DAG:         [[VAR_v011_:%.+]] = arith.subf [[VAR_v007_]], [[VAR_v008_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.cmpf oge, [[VAR_v011_]], [[CST_5_dot_000000_1_]] : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v013_:%.+]] = arith.addi [[VAR_v010_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v014_:%.+]] = arith.select [[VAR_v012_]], [[VAR_v013_]], [[VAR_v010_]] : index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_v015_:%.+]] = arith.cmpi slt, [[VAR_v014_]], [[CST_0_]] : index
// CHECK:             [[VAR_v016_:%.+]] = arith.select [[VAR_v015_]], [[CST_0_]], [[VAR_v014_]] : index
// CHECK:             [[VAR_v017_:%.+]] = arith.cmpi sgt, [[VAR_v016_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_v018_:%.+]] = arith.select [[VAR_v017_]], [[CST_0_1_]], [[VAR_v016_]] : index
// CHECK-DAG:         [[VAR_v019_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.index_cast [[VAR_v019_]] : index to i64
// CHECK-DAG:         [[VAR_v021_:%.+]] = arith.sitofp [[VAR_v020_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_2_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v022_:%.+]] = arith.addf [[VAR_v021_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_v023_:%.+]] = arith.divf [[VAR_v022_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v024_:%.+]] = arith.subf [[VAR_v023_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_v025_:%.+]] = math.floor [[VAR_v024_]] : f32
// CHECK:             [[VAR_v026_:%.+]] = arith.fptosi [[VAR_v025_]] : f32 to i64
// CHECK-DAG:         [[VAR_v027_:%.+]] = arith.index_cast [[VAR_v026_]] : i64 to index
// CHECK-DAG:         [[VAR_v028_:%.+]] = arith.subf [[VAR_v024_]], [[VAR_v025_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_3_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v029_:%.+]] = arith.cmpf oge, [[VAR_v028_]], [[CST_5_dot_000000_3_]] : f32
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v030_:%.+]] = arith.addi [[VAR_v027_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v031_:%.+]] = arith.select [[VAR_v029_]], [[VAR_v030_]], [[VAR_v027_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v032_:%.+]] = arith.cmpi slt, [[VAR_v031_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v033_:%.+]] = arith.select [[VAR_v032_]], [[CST_0_2_]], [[VAR_v031_]] : index
// CHECK:             [[VAR_v034_:%.+]] = arith.cmpi sgt, [[VAR_v033_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_v035_:%.+]] = arith.select [[VAR_v034_]], [[CST_1_2_]], [[VAR_v033_]] : index
// CHECK-DAG:         [[VAR_v036_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v037_:%.+]] = arith.index_cast [[VAR_v036_]] : index to i64
// CHECK-DAG:         [[VAR_v038_:%.+]] = arith.sitofp [[VAR_v037_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_4_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v039_:%.+]] = arith.addf [[VAR_v038_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_v040_:%.+]] = arith.divf [[VAR_v039_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v041_:%.+]] = arith.subf [[VAR_v040_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_v042_:%.+]] = math.floor [[VAR_v041_]] : f32
// CHECK:             [[VAR_v043_:%.+]] = arith.fptosi [[VAR_v042_]] : f32 to i64
// CHECK-DAG:         [[VAR_v044_:%.+]] = arith.index_cast [[VAR_v043_]] : i64 to index
// CHECK-DAG:         [[VAR_v045_:%.+]] = arith.subf [[VAR_v041_]], [[VAR_v042_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_5_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v046_:%.+]] = arith.cmpf oge, [[VAR_v045_]], [[CST_5_dot_000000_5_]] : f32
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v047_:%.+]] = arith.addi [[VAR_v044_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_v048_:%.+]] = arith.select [[VAR_v046_]], [[VAR_v047_]], [[VAR_v044_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v049_:%.+]] = arith.cmpi slt, [[VAR_v048_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v050_:%.+]] = arith.select [[VAR_v049_]], [[CST_0_3_]], [[VAR_v048_]] : index
// CHECK:             [[VAR_v051_:%.+]] = arith.cmpi sgt, [[VAR_v050_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_v052_:%.+]] = arith.select [[VAR_v051_]], [[CST_1_4_]], [[VAR_v050_]] : index
// CHECK-DAG:         [[VAR_v053_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v054_:%.+]] = arith.index_cast [[VAR_v053_]] : index to i64
// CHECK-DAG:         [[VAR_v055_:%.+]] = arith.sitofp [[VAR_v054_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_6_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v056_:%.+]] = arith.addf [[VAR_v055_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_v057_:%.+]] = arith.divf [[VAR_v056_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v058_:%.+]] = arith.subf [[VAR_v057_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_v059_:%.+]] = math.floor [[VAR_v058_]] : f32
// CHECK:             [[VAR_v060_:%.+]] = arith.fptosi [[VAR_v059_]] : f32 to i64
// CHECK-DAG:         [[VAR_v061_:%.+]] = arith.index_cast [[VAR_v060_]] : i64 to index
// CHECK-DAG:         [[VAR_v062_:%.+]] = arith.subf [[VAR_v058_]], [[VAR_v059_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_7_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v063_:%.+]] = arith.cmpf oge, [[VAR_v062_]], [[CST_5_dot_000000_7_]] : f32
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v064_:%.+]] = arith.addi [[VAR_v061_]], [[CST_1_5_]] : index
// CHECK-DAG:         [[VAR_v065_:%.+]] = arith.select [[VAR_v063_]], [[VAR_v064_]], [[VAR_v061_]] : index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v066_:%.+]] = arith.cmpi slt, [[VAR_v065_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_v067_:%.+]] = arith.select [[VAR_v066_]], [[CST_0_4_]], [[VAR_v065_]] : index
// CHECK:             [[VAR_v068_:%.+]] = arith.cmpi sgt, [[VAR_v067_]], [[CST_1_6_]] : index
// CHECK:             [[VAR_v069_:%.+]] = arith.select [[VAR_v068_]], [[CST_1_6_]], [[VAR_v067_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v018_]], [[VAR_v035_]], [[VAR_v052_]], [[VAR_v069_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_v001_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_nearest_half_pixel_round_prefer_floor(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 0 : i64, nearest_mode = 2 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_half_pixel_round_prefer_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK:             [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_v003_:%.+]] = arith.index_cast [[VAR_v002_]] : index to i64
// CHECK-DAG:         [[VAR_v004_:%.+]] = arith.sitofp [[VAR_v003_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v005_:%.+]] = arith.addf [[VAR_v004_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v006_:%.+]] = arith.divf [[VAR_v005_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.subf [[VAR_v006_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v008_:%.+]] = math.floor [[VAR_v007_]] : f32
// CHECK:             [[VAR_v009_:%.+]] = arith.fptosi [[VAR_v008_]] : f32 to i64
// CHECK-DAG:         [[VAR_v010_:%.+]] = arith.index_cast [[VAR_v009_]] : i64 to index
// CHECK-DAG:         [[VAR_v011_:%.+]] = arith.subf [[VAR_v007_]], [[VAR_v008_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.cmpf ogt, [[VAR_v011_]], [[CST_5_dot_000000_1_]] : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v013_:%.+]] = arith.addi [[VAR_v010_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v014_:%.+]] = arith.select [[VAR_v012_]], [[VAR_v013_]], [[VAR_v010_]] : index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_v015_:%.+]] = arith.cmpi slt, [[VAR_v014_]], [[CST_0_]] : index
// CHECK:             [[VAR_v016_:%.+]] = arith.select [[VAR_v015_]], [[CST_0_]], [[VAR_v014_]] : index
// CHECK:             [[VAR_v017_:%.+]] = arith.cmpi sgt, [[VAR_v016_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_v018_:%.+]] = arith.select [[VAR_v017_]], [[CST_0_1_]], [[VAR_v016_]] : index
// CHECK-DAG:         [[VAR_v019_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.index_cast [[VAR_v019_]] : index to i64
// CHECK-DAG:         [[VAR_v021_:%.+]] = arith.sitofp [[VAR_v020_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_2_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v022_:%.+]] = arith.addf [[VAR_v021_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_v023_:%.+]] = arith.divf [[VAR_v022_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v024_:%.+]] = arith.subf [[VAR_v023_]], [[CST_5_dot_000000_2_]] : f32
// CHECK:             [[VAR_v025_:%.+]] = math.floor [[VAR_v024_]] : f32
// CHECK:             [[VAR_v026_:%.+]] = arith.fptosi [[VAR_v025_]] : f32 to i64
// CHECK-DAG:         [[VAR_v027_:%.+]] = arith.index_cast [[VAR_v026_]] : i64 to index
// CHECK-DAG:         [[VAR_v028_:%.+]] = arith.subf [[VAR_v024_]], [[VAR_v025_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_3_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v029_:%.+]] = arith.cmpf ogt, [[VAR_v028_]], [[CST_5_dot_000000_3_]] : f32
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v030_:%.+]] = arith.addi [[VAR_v027_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v031_:%.+]] = arith.select [[VAR_v029_]], [[VAR_v030_]], [[VAR_v027_]] : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v032_:%.+]] = arith.cmpi slt, [[VAR_v031_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v033_:%.+]] = arith.select [[VAR_v032_]], [[CST_0_2_]], [[VAR_v031_]] : index
// CHECK:             [[VAR_v034_:%.+]] = arith.cmpi sgt, [[VAR_v033_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_v035_:%.+]] = arith.select [[VAR_v034_]], [[CST_1_2_]], [[VAR_v033_]] : index
// CHECK-DAG:         [[VAR_v036_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v037_:%.+]] = arith.index_cast [[VAR_v036_]] : index to i64
// CHECK-DAG:         [[VAR_v038_:%.+]] = arith.sitofp [[VAR_v037_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_4_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v039_:%.+]] = arith.addf [[VAR_v038_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_v040_:%.+]] = arith.divf [[VAR_v039_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v041_:%.+]] = arith.subf [[VAR_v040_]], [[CST_5_dot_000000_4_]] : f32
// CHECK:             [[VAR_v042_:%.+]] = math.floor [[VAR_v041_]] : f32
// CHECK:             [[VAR_v043_:%.+]] = arith.fptosi [[VAR_v042_]] : f32 to i64
// CHECK-DAG:         [[VAR_v044_:%.+]] = arith.index_cast [[VAR_v043_]] : i64 to index
// CHECK-DAG:         [[VAR_v045_:%.+]] = arith.subf [[VAR_v041_]], [[VAR_v042_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_5_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v046_:%.+]] = arith.cmpf ogt, [[VAR_v045_]], [[CST_5_dot_000000_5_]] : f32
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v047_:%.+]] = arith.addi [[VAR_v044_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_v048_:%.+]] = arith.select [[VAR_v046_]], [[VAR_v047_]], [[VAR_v044_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v049_:%.+]] = arith.cmpi slt, [[VAR_v048_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v050_:%.+]] = arith.select [[VAR_v049_]], [[CST_0_3_]], [[VAR_v048_]] : index
// CHECK:             [[VAR_v051_:%.+]] = arith.cmpi sgt, [[VAR_v050_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_v052_:%.+]] = arith.select [[VAR_v051_]], [[CST_1_4_]], [[VAR_v050_]] : index
// CHECK-DAG:         [[VAR_v053_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v054_:%.+]] = arith.index_cast [[VAR_v053_]] : index to i64
// CHECK-DAG:         [[VAR_v055_:%.+]] = arith.sitofp [[VAR_v054_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_6_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v056_:%.+]] = arith.addf [[VAR_v055_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_v057_:%.+]] = arith.divf [[VAR_v056_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v058_:%.+]] = arith.subf [[VAR_v057_]], [[CST_5_dot_000000_6_]] : f32
// CHECK:             [[VAR_v059_:%.+]] = math.floor [[VAR_v058_]] : f32
// CHECK:             [[VAR_v060_:%.+]] = arith.fptosi [[VAR_v059_]] : f32 to i64
// CHECK-DAG:         [[VAR_v061_:%.+]] = arith.index_cast [[VAR_v060_]] : i64 to index
// CHECK-DAG:         [[VAR_v062_:%.+]] = arith.subf [[VAR_v058_]], [[VAR_v059_]] : f32
// CHECK-DAG:         [[CST_5_dot_000000_7_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v063_:%.+]] = arith.cmpf ogt, [[VAR_v062_]], [[CST_5_dot_000000_7_]] : f32
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v064_:%.+]] = arith.addi [[VAR_v061_]], [[CST_1_5_]] : index
// CHECK-DAG:         [[VAR_v065_:%.+]] = arith.select [[VAR_v063_]], [[VAR_v064_]], [[VAR_v061_]] : index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v066_:%.+]] = arith.cmpi slt, [[VAR_v065_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_v067_:%.+]] = arith.select [[VAR_v066_]], [[CST_0_4_]], [[VAR_v065_]] : index
// CHECK:             [[VAR_v068_:%.+]] = arith.cmpi sgt, [[VAR_v067_]], [[CST_1_6_]] : index
// CHECK:             [[VAR_v069_:%.+]] = arith.select [[VAR_v068_]], [[CST_1_6_]], [[VAR_v067_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v018_]], [[VAR_v035_]], [[VAR_v052_]], [[VAR_v069_]]{{.}} : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield [[VAR_extracted_]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return [[VAR_v001_]] : tensor<1x2x4x4xf32>
// CHECK:         }

// -----

func.func @resize_linear_half_pixel(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 0 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_linear_half_pixel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x1x4x4xf32>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x1x4x4xf32>) {
// CHECK:           ^bb0([[OUT_:%.+]]: f32):
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK-DAG:         [[VAR_v003_:%.+]] = linalg.index 1 : index
// CHECK-DAG:         [[VAR_v004_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v005_:%.+]] = arith.index_cast [[VAR_v004_]] : index to i64
// CHECK-DAG:         [[VAR_v006_:%.+]] = arith.sitofp [[VAR_v005_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.addf [[VAR_v006_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v008_:%.+]] = arith.divf [[VAR_v007_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v009_:%.+]] = arith.subf [[VAR_v008_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_v010_:%.+]] = math.floor [[VAR_v009_]] : f32
// CHECK:             [[VAR_v011_:%.+]] = arith.fptosi [[VAR_v010_]] : f32 to i64
// CHECK:             [[VAR_v012_:%.+]] = arith.index_cast [[VAR_v011_]] : i64 to index
// CHECK-DAG:         [[VAR_v013_:%.+]] = arith.addi [[VAR_v012_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v014_:%.+]] = arith.subf [[VAR_v009_]], [[VAR_v010_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v015_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_v014_]] : f32
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v016_:%.+]] = arith.cmpi slt, [[VAR_v012_]], [[CST_0_]] : index
// CHECK:             [[VAR_v017_:%.+]] = arith.select [[VAR_v016_]], [[CST_0_]], [[VAR_v012_]] : index
// CHECK:             [[VAR_v018_:%.+]] = arith.cmpi sgt, [[VAR_v017_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v019_:%.+]] = arith.select [[VAR_v018_]], [[CST_1_1_]], [[VAR_v017_]] : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.cmpi slt, [[VAR_v013_]], [[CST_0_1_]] : index
// CHECK:             [[VAR_v021_:%.+]] = arith.select [[VAR_v020_]], [[CST_0_1_]], [[VAR_v013_]] : index
// CHECK:             [[VAR_v022_:%.+]] = arith.cmpi sgt, [[VAR_v021_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_v023_:%.+]] = arith.select [[VAR_v022_]], [[CST_1_2_]], [[VAR_v021_]] : index
// CHECK-DAG:         [[VAR_v024_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v025_:%.+]] = arith.index_cast [[VAR_v024_]] : index to i64
// CHECK-DAG:         [[VAR_v026_:%.+]] = arith.sitofp [[VAR_v025_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_5_dot_000000_1_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:             [[VAR_v027_:%.+]] = arith.addf [[VAR_v026_]], [[CST_5_dot_000000_1_]] : f32
// CHECK:             [[VAR_v028_:%.+]] = arith.divf [[VAR_v027_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v029_:%.+]] = arith.subf [[VAR_v028_]], [[CST_5_dot_000000_1_]] : f32
// CHECK:             [[VAR_v030_:%.+]] = math.floor [[VAR_v029_]] : f32
// CHECK:             [[VAR_v031_:%.+]] = arith.fptosi [[VAR_v030_]] : f32 to i64
// CHECK:             [[VAR_v032_:%.+]] = arith.index_cast [[VAR_v031_]] : i64 to index
// CHECK-DAG:         [[VAR_v033_:%.+]] = arith.addi [[VAR_v032_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v034_:%.+]] = arith.subf [[VAR_v029_]], [[VAR_v030_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v035_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_v034_]] : f32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v036_:%.+]] = arith.cmpi slt, [[VAR_v032_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v037_:%.+]] = arith.select [[VAR_v036_]], [[CST_0_2_]], [[VAR_v032_]] : index
// CHECK:             [[VAR_v038_:%.+]] = arith.cmpi sgt, [[VAR_v037_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_v039_:%.+]] = arith.select [[VAR_v038_]], [[CST_1_3_]], [[VAR_v037_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v040_:%.+]] = arith.cmpi slt, [[VAR_v033_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v041_:%.+]] = arith.select [[VAR_v040_]], [[CST_0_3_]], [[VAR_v033_]] : index
// CHECK:             [[VAR_v042_:%.+]] = arith.cmpi sgt, [[VAR_v041_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_v043_:%.+]] = arith.select [[VAR_v042_]], [[CST_1_4_]], [[VAR_v041_]] : index
// CHECK-DAG:         [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v044_:%.+]] = arith.mulf [[CST_1_dot_000000_1_]], [[VAR_v015_]] : f32
// CHECK-DAG:         [[VAR_v045_:%.+]] = arith.mulf [[VAR_v044_]], [[VAR_v035_]] : f32
// CHECK-DAG:         [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v019_]], [[VAR_v039_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_v046_:%.+]] = arith.mulf [[VAR_extracted_]], [[VAR_v045_]] : f32
// CHECK-DAG:         [[VAR_v047_:%.+]] = arith.addf [[CST_0_dot_000000_]], [[VAR_v046_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_2_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v048_:%.+]] = arith.mulf [[CST_1_dot_000000_2_]], [[VAR_v014_]] : f32
// CHECK-DAG:         [[VAR_v049_:%.+]] = arith.mulf [[VAR_v048_]], [[VAR_v035_]] : f32
// CHECK-DAG:         [[VAR_extracted_14_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v023_]], [[VAR_v039_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_v050_:%.+]] = arith.mulf [[VAR_extracted_14_]], [[VAR_v049_]] : f32
// CHECK-DAG:         [[VAR_v051_:%.+]] = arith.addf [[VAR_v047_]], [[VAR_v050_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_3_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v052_:%.+]] = arith.mulf [[CST_1_dot_000000_3_]], [[VAR_v015_]] : f32
// CHECK-DAG:         [[VAR_v053_:%.+]] = arith.mulf [[VAR_v052_]], [[VAR_v034_]] : f32
// CHECK-DAG:         [[VAR_extracted_16_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v019_]], [[VAR_v043_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_v054_:%.+]] = arith.mulf [[VAR_extracted_16_]], [[VAR_v053_]] : f32
// CHECK-DAG:         [[VAR_v055_:%.+]] = arith.addf [[VAR_v051_]], [[VAR_v054_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_4_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v056_:%.+]] = arith.mulf [[CST_1_dot_000000_4_]], [[VAR_v014_]] : f32
// CHECK-DAG:         [[VAR_v057_:%.+]] = arith.mulf [[VAR_v056_]], [[VAR_v034_]] : f32
// CHECK-DAG:         [[VAR_extracted_18_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v023_]], [[VAR_v043_]]{{.}} : tensor<1x1x2x2xf32>
// CHECK:             [[VAR_v058_:%.+]] = arith.mulf [[VAR_extracted_18_]], [[VAR_v057_]] : f32
// CHECK:             [[VAR_v059_:%.+]] = arith.addf [[VAR_v055_]], [[VAR_v058_]] : f32
// CHECK:             linalg.yield [[VAR_v059_]] : f32
// CHECK:           } -> tensor<1x1x4x4xf32>
// CHECK:           return [[VAR_v001_]] : tensor<1x1x4x4xf32>
// CHECK:         }

// -----

func.func @resize_linear_align_corners(%arg0: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 3 : i64, mode = 1 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16>
  return %0 : tensor<1x1x4x4xbf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_linear_align_corners
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x1x4x4xbf16>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x1x4x4xbf16>) {
// CHECK:           ^bb0([[OUT_:%.+]]: bf16):
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK-DAG:         [[VAR_v003_:%.+]] = linalg.index 1 : index
// CHECK-DAG:         [[VAR_v004_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v005_:%.+]] = arith.index_cast [[VAR_v004_]] : index to i64
// CHECK-DAG:         [[VAR_v006_:%.+]] = arith.sitofp [[VAR_v005_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.mulf [[VAR_v006_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v008_:%.+]] = arith.divf [[VAR_v007_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_v009_:%.+]] = math.floor [[VAR_v008_]] : f32
// CHECK:             [[VAR_v010_:%.+]] = arith.fptosi [[VAR_v009_]] : f32 to i64
// CHECK:             [[VAR_v011_:%.+]] = arith.index_cast [[VAR_v010_]] : i64 to index
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.addi [[VAR_v011_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v013_:%.+]] = arith.subf [[VAR_v008_]], [[VAR_v009_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v014_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_v013_]] : f32
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v015_:%.+]] = arith.cmpi slt, [[VAR_v011_]], [[CST_0_]] : index
// CHECK:             [[VAR_v016_:%.+]] = arith.select [[VAR_v015_]], [[CST_0_]], [[VAR_v011_]] : index
// CHECK:             [[VAR_v017_:%.+]] = arith.cmpi sgt, [[VAR_v016_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v018_:%.+]] = arith.select [[VAR_v017_]], [[CST_1_1_]], [[VAR_v016_]] : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v019_:%.+]] = arith.cmpi slt, [[VAR_v012_]], [[CST_0_1_]] : index
// CHECK:             [[VAR_v020_:%.+]] = arith.select [[VAR_v019_]], [[CST_0_1_]], [[VAR_v012_]] : index
// CHECK:             [[VAR_v021_:%.+]] = arith.cmpi sgt, [[VAR_v020_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_v022_:%.+]] = arith.select [[VAR_v021_]], [[CST_1_2_]], [[VAR_v020_]] : index
// CHECK-DAG:         [[VAR_v023_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v024_:%.+]] = arith.index_cast [[VAR_v023_]] : index to i64
// CHECK-DAG:         [[VAR_v025_:%.+]] = arith.sitofp [[VAR_v024_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_2_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_3_dot_000000_1_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:             [[VAR_v026_:%.+]] = arith.mulf [[VAR_v025_]], [[CST_1_dot_000000_2_]] : f32
// CHECK:             [[VAR_v027_:%.+]] = arith.divf [[VAR_v026_]], [[CST_3_dot_000000_1_]] : f32
// CHECK:             [[VAR_v028_:%.+]] = math.floor [[VAR_v027_]] : f32
// CHECK:             [[VAR_v029_:%.+]] = arith.fptosi [[VAR_v028_]] : f32 to i64
// CHECK:             [[VAR_v030_:%.+]] = arith.index_cast [[VAR_v029_]] : i64 to index
// CHECK-DAG:         [[VAR_v031_:%.+]] = arith.addi [[VAR_v030_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v032_:%.+]] = arith.subf [[VAR_v027_]], [[VAR_v028_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_v033_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_v032_]] : f32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v034_:%.+]] = arith.cmpi slt, [[VAR_v030_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v035_:%.+]] = arith.select [[VAR_v034_]], [[CST_0_2_]], [[VAR_v030_]] : index
// CHECK:             [[VAR_v036_:%.+]] = arith.cmpi sgt, [[VAR_v035_]], [[CST_1_3_]] : index
// CHECK-DAG:         [[VAR_v037_:%.+]] = arith.select [[VAR_v036_]], [[CST_1_3_]], [[VAR_v035_]] : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v038_:%.+]] = arith.cmpi slt, [[VAR_v031_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v039_:%.+]] = arith.select [[VAR_v038_]], [[CST_0_3_]], [[VAR_v031_]] : index
// CHECK:             [[VAR_v040_:%.+]] = arith.cmpi sgt, [[VAR_v039_]], [[CST_1_4_]] : index
// CHECK-DAG:         [[VAR_v041_:%.+]] = arith.select [[VAR_v040_]], [[CST_1_4_]], [[VAR_v039_]] : index
// CHECK-DAG:         [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         [[CST_1_dot_000000_3_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v042_:%.+]] = arith.mulf [[CST_1_dot_000000_3_]], [[VAR_v014_]] : f32
// CHECK-DAG:         [[VAR_v043_:%.+]] = arith.mulf [[VAR_v042_]], [[VAR_v033_]] : f32
// CHECK-DAG:         [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v018_]], [[VAR_v037_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_v044_:%.+]] = arith.extf [[VAR_extracted_]] : bf16 to f32
// CHECK:             [[VAR_v045_:%.+]] = arith.mulf [[VAR_v044_]], [[VAR_v043_]] : f32
// CHECK-DAG:         [[VAR_v046_:%.+]] = arith.addf [[CST_0_dot_000000_]], [[VAR_v045_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_4_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v047_:%.+]] = arith.mulf [[CST_1_dot_000000_4_]], [[VAR_v013_]] : f32
// CHECK-DAG:         [[VAR_v048_:%.+]] = arith.mulf [[VAR_v047_]], [[VAR_v033_]] : f32
// CHECK-DAG:         [[VAR_extracted_16_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v022_]], [[VAR_v037_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_v049_:%.+]] = arith.extf [[VAR_extracted_16_]] : bf16 to f32
// CHECK:             [[VAR_v050_:%.+]] = arith.mulf [[VAR_v049_]], [[VAR_v048_]] : f32
// CHECK-DAG:         [[VAR_v051_:%.+]] = arith.addf [[VAR_v046_]], [[VAR_v050_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_5_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v052_:%.+]] = arith.mulf [[CST_1_dot_000000_5_]], [[VAR_v014_]] : f32
// CHECK-DAG:         [[VAR_v053_:%.+]] = arith.mulf [[VAR_v052_]], [[VAR_v032_]] : f32
// CHECK-DAG:         [[VAR_extracted_18_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v018_]], [[VAR_v041_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_v054_:%.+]] = arith.extf [[VAR_extracted_18_]] : bf16 to f32
// CHECK:             [[VAR_v055_:%.+]] = arith.mulf [[VAR_v054_]], [[VAR_v053_]] : f32
// CHECK-DAG:         [[VAR_v056_:%.+]] = arith.addf [[VAR_v051_]], [[VAR_v055_]] : f32
// CHECK-DAG:         [[CST_1_dot_000000_6_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v057_:%.+]] = arith.mulf [[CST_1_dot_000000_6_]], [[VAR_v013_]] : f32
// CHECK-DAG:         [[VAR_v058_:%.+]] = arith.mulf [[VAR_v057_]], [[VAR_v032_]] : f32
// CHECK-DAG:         [[VAR_extracted_20_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v002_]], [[VAR_v003_]], [[VAR_v022_]], [[VAR_v041_]]{{.}} : tensor<1x1x2x2xbf16>
// CHECK:             [[VAR_v059_:%.+]] = arith.extf [[VAR_extracted_20_]] : bf16 to f32
// CHECK:             [[VAR_v060_:%.+]] = arith.mulf [[VAR_v059_]], [[VAR_v058_]] : f32
// CHECK:             [[VAR_v061_:%.+]] = arith.addf [[VAR_v056_]], [[VAR_v060_]] : f32
// CHECK:             [[VAR_v062_:%.+]] = arith.truncf [[VAR_v061_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_v062_]] : bf16
// CHECK:           } -> tensor<1x1x4x4xbf16>
// CHECK:           return [[VAR_v001_]] : tensor<1x1x4x4xbf16>
// CHECK:         }

// -----

func.func @resize_nearest_5d(%arg0: tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0, 2.0>} : (tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16>
  return %0 : tensor<2x16x32x8x8xbf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL:  func.func @resize_nearest_5d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x16x16x4x4xbf16>) -> tensor<2x16x32x8x8xbf16> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<2x16x32x8x8xbf16>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<2x16x32x8x8xbf16>) {
// CHECK:           ^bb0([[OUT_:%.+]]: bf16):
// CHECK:             [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_v003_:%.+]] = arith.index_cast [[VAR_v002_]] : index to i64
// CHECK-DAG:         [[VAR_v004_:%.+]] = arith.sitofp [[VAR_v003_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v005_:%.+]] = arith.divf [[VAR_v004_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_v006_:%.+]] = math.floor [[VAR_v005_]] : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.fptosi [[VAR_v006_]] : f32 to i64
// CHECK-DAG:         [[VAR_v008_:%.+]] = arith.index_cast [[VAR_v007_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v009_:%.+]] = arith.cmpi slt, [[VAR_v008_]], [[CST_0_]] : index
// CHECK:             [[VAR_v010_:%.+]] = arith.select [[VAR_v009_]], [[CST_0_]], [[VAR_v008_]] : index
// CHECK:             [[VAR_v011_:%.+]] = arith.cmpi sgt, [[VAR_v010_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.select [[VAR_v011_]], [[CST_1_]], [[VAR_v010_]] : index
// CHECK-DAG:         [[VAR_v013_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_v014_:%.+]] = arith.index_cast [[VAR_v013_]] : index to i64
// CHECK-DAG:         [[VAR_v015_:%.+]] = arith.sitofp [[VAR_v014_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v016_:%.+]] = arith.divf [[VAR_v015_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v017_:%.+]] = math.floor [[VAR_v016_]] : f32
// CHECK:             [[VAR_v018_:%.+]] = arith.fptosi [[VAR_v017_]] : f32 to i64
// CHECK-DAG:         [[VAR_v019_:%.+]] = arith.index_cast [[VAR_v018_]] : i64 to index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_15_:%.+]] = arith.constant 15 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.cmpi slt, [[VAR_v019_]], [[CST_0_1_]] : index
// CHECK:             [[VAR_v021_:%.+]] = arith.select [[VAR_v020_]], [[CST_0_1_]], [[VAR_v019_]] : index
// CHECK:             [[VAR_v022_:%.+]] = arith.cmpi sgt, [[VAR_v021_]], [[CST_15_]] : index
// CHECK-DAG:         [[VAR_v023_:%.+]] = arith.select [[VAR_v022_]], [[CST_15_]], [[VAR_v021_]] : index
// CHECK-DAG:         [[VAR_v024_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v025_:%.+]] = arith.index_cast [[VAR_v024_]] : index to i64
// CHECK-DAG:         [[VAR_v026_:%.+]] = arith.sitofp [[VAR_v025_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v027_:%.+]] = arith.divf [[VAR_v026_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v028_:%.+]] = math.floor [[VAR_v027_]] : f32
// CHECK:             [[VAR_v029_:%.+]] = arith.fptosi [[VAR_v028_]] : f32 to i64
// CHECK-DAG:         [[VAR_v030_:%.+]] = arith.index_cast [[VAR_v029_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_15_1_:%.+]] = arith.constant 15 : index
// CHECK:             [[VAR_v031_:%.+]] = arith.cmpi slt, [[VAR_v030_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v032_:%.+]] = arith.select [[VAR_v031_]], [[CST_0_2_]], [[VAR_v030_]] : index
// CHECK:             [[VAR_v033_:%.+]] = arith.cmpi sgt, [[VAR_v032_]], [[CST_15_1_]] : index
// CHECK-DAG:         [[VAR_v034_:%.+]] = arith.select [[VAR_v033_]], [[CST_15_1_]], [[VAR_v032_]] : index
// CHECK-DAG:         [[VAR_v035_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v036_:%.+]] = arith.index_cast [[VAR_v035_]] : index to i64
// CHECK-DAG:         [[VAR_v037_:%.+]] = arith.sitofp [[VAR_v036_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v038_:%.+]] = arith.divf [[VAR_v037_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v039_:%.+]] = math.floor [[VAR_v038_]] : f32
// CHECK:             [[VAR_v040_:%.+]] = arith.fptosi [[VAR_v039_]] : f32 to i64
// CHECK-DAG:         [[VAR_v041_:%.+]] = arith.index_cast [[VAR_v040_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:             [[VAR_v042_:%.+]] = arith.cmpi slt, [[VAR_v041_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v043_:%.+]] = arith.select [[VAR_v042_]], [[CST_0_3_]], [[VAR_v041_]] : index
// CHECK:             [[VAR_v044_:%.+]] = arith.cmpi sgt, [[VAR_v043_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_v045_:%.+]] = arith.select [[VAR_v044_]], [[CST_3_]], [[VAR_v043_]] : index
// CHECK-DAG:         [[VAR_v046_:%.+]] = linalg.index 4 : index
// CHECK:             [[VAR_v047_:%.+]] = arith.index_cast [[VAR_v046_]] : index to i64
// CHECK-DAG:         [[VAR_v048_:%.+]] = arith.sitofp [[VAR_v047_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_2_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v049_:%.+]] = arith.divf [[VAR_v048_]], [[CST_2_dot_000000_2_]] : f32
// CHECK:             [[VAR_v050_:%.+]] = math.floor [[VAR_v049_]] : f32
// CHECK:             [[VAR_v051_:%.+]] = arith.fptosi [[VAR_v050_]] : f32 to i64
// CHECK-DAG:         [[VAR_v052_:%.+]] = arith.index_cast [[VAR_v051_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK:             [[VAR_v053_:%.+]] = arith.cmpi slt, [[VAR_v052_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_v054_:%.+]] = arith.select [[VAR_v053_]], [[CST_0_4_]], [[VAR_v052_]] : index
// CHECK:             [[VAR_v055_:%.+]] = arith.cmpi sgt, [[VAR_v054_]], [[CST_3_1_]] : index
// CHECK:             [[VAR_v056_:%.+]] = arith.select [[VAR_v055_]], [[CST_3_1_]], [[VAR_v054_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v012_]], [[VAR_v023_]], [[VAR_v034_]], [[VAR_v045_]], [[VAR_v056_]]{{.}} : tensor<2x16x16x4x4xbf16>
// CHECK:             linalg.yield [[VAR_extracted_]] : bf16
// CHECK:           } -> tensor<2x16x32x8x8xbf16>
// CHECK:           return [[VAR_v001_]] : tensor<2x16x32x8x8xbf16>
// CHECK:         }

// -----

func.func @resize_nearest_i8(%arg0: tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8> {
  %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8>
  return %0 : tensor<1x2x4x4xi8>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:  func.func @resize_nearest_i8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2x2xi8>) -> tensor<1x2x4x4xi8> {
// CHECK:           [[VAR_v000_:%.+]] = tensor.empty() : tensor<1x2x4x4xi8>
// CHECK:           [[VAR_v001_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[VAR_v000_]] : tensor<1x2x4x4xi8>) {
// CHECK:           ^bb0([[OUT_:%.+]]: i8):
// CHECK:             [[VAR_v002_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_v003_:%.+]] = arith.index_cast [[VAR_v002_]] : index to i64
// CHECK-DAG:         [[VAR_v004_:%.+]] = arith.sitofp [[VAR_v003_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v005_:%.+]] = arith.divf [[VAR_v004_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_v006_:%.+]] = math.floor [[VAR_v005_]] : f32
// CHECK:             [[VAR_v007_:%.+]] = arith.fptosi [[VAR_v006_]] : f32 to i64
// CHECK-DAG:         [[VAR_v008_:%.+]] = arith.index_cast [[VAR_v007_]] : i64 to index
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_v009_:%.+]] = arith.cmpi slt, [[VAR_v008_]], [[CST_0_]] : index
// CHECK:             [[VAR_v010_:%.+]] = arith.select [[VAR_v009_]], [[CST_0_]], [[VAR_v008_]] : index
// CHECK:             [[VAR_v011_:%.+]] = arith.cmpi sgt, [[VAR_v010_]], [[CST_0_1_]] : index
// CHECK-DAG:         [[VAR_v012_:%.+]] = arith.select [[VAR_v011_]], [[CST_0_1_]], [[VAR_v010_]] : index
// CHECK-DAG:         [[VAR_v013_:%.+]] = linalg.index 1 : index
// CHECK:             [[VAR_v014_:%.+]] = arith.index_cast [[VAR_v013_]] : index to i64
// CHECK-DAG:         [[VAR_v015_:%.+]] = arith.sitofp [[VAR_v014_]] : i64 to f32
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK:             [[VAR_v016_:%.+]] = arith.divf [[VAR_v015_]], [[CST_1_dot_000000_1_]] : f32
// CHECK:             [[VAR_v017_:%.+]] = math.floor [[VAR_v016_]] : f32
// CHECK:             [[VAR_v018_:%.+]] = arith.fptosi [[VAR_v017_]] : f32 to i64
// CHECK-DAG:         [[VAR_v019_:%.+]] = arith.index_cast [[VAR_v018_]] : i64 to index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v020_:%.+]] = arith.cmpi slt, [[VAR_v019_]], [[CST_0_2_]] : index
// CHECK:             [[VAR_v021_:%.+]] = arith.select [[VAR_v020_]], [[CST_0_2_]], [[VAR_v019_]] : index
// CHECK:             [[VAR_v022_:%.+]] = arith.cmpi sgt, [[VAR_v021_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_v023_:%.+]] = arith.select [[VAR_v022_]], [[CST_1_]], [[VAR_v021_]] : index
// CHECK-DAG:         [[VAR_v024_:%.+]] = linalg.index 2 : index
// CHECK:             [[VAR_v025_:%.+]] = arith.index_cast [[VAR_v024_]] : index to i64
// CHECK-DAG:         [[VAR_v026_:%.+]] = arith.sitofp [[VAR_v025_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v027_:%.+]] = arith.divf [[VAR_v026_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_v028_:%.+]] = math.floor [[VAR_v027_]] : f32
// CHECK:             [[VAR_v029_:%.+]] = arith.fptosi [[VAR_v028_]] : f32 to i64
// CHECK-DAG:         [[VAR_v030_:%.+]] = arith.index_cast [[VAR_v029_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v031_:%.+]] = arith.cmpi slt, [[VAR_v030_]], [[CST_0_3_]] : index
// CHECK:             [[VAR_v032_:%.+]] = arith.select [[VAR_v031_]], [[CST_0_3_]], [[VAR_v030_]] : index
// CHECK:             [[VAR_v033_:%.+]] = arith.cmpi sgt, [[VAR_v032_]], [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_v034_:%.+]] = arith.select [[VAR_v033_]], [[CST_1_1_]], [[VAR_v032_]] : index
// CHECK-DAG:         [[VAR_v035_:%.+]] = linalg.index 3 : index
// CHECK:             [[VAR_v036_:%.+]] = arith.index_cast [[VAR_v035_]] : index to i64
// CHECK-DAG:         [[VAR_v037_:%.+]] = arith.sitofp [[VAR_v036_]] : i64 to f32
// CHECK-DAG:         [[CST_2_dot_000000_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:             [[VAR_v038_:%.+]] = arith.divf [[VAR_v037_]], [[CST_2_dot_000000_1_]] : f32
// CHECK:             [[VAR_v039_:%.+]] = math.floor [[VAR_v038_]] : f32
// CHECK:             [[VAR_v040_:%.+]] = arith.fptosi [[VAR_v039_]] : f32 to i64
// CHECK-DAG:         [[VAR_v041_:%.+]] = arith.index_cast [[VAR_v040_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_v042_:%.+]] = arith.cmpi slt, [[VAR_v041_]], [[CST_0_4_]] : index
// CHECK:             [[VAR_v043_:%.+]] = arith.select [[VAR_v042_]], [[CST_0_4_]], [[VAR_v041_]] : index
// CHECK:             [[VAR_v044_:%.+]] = arith.cmpi sgt, [[VAR_v043_]], [[CST_1_2_]] : index
// CHECK:             [[VAR_v045_:%.+]] = arith.select [[VAR_v044_]], [[CST_1_2_]], [[VAR_v043_]] : index
// CHECK:             [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_0_]]{{.}}[[VAR_v012_]], [[VAR_v023_]], [[VAR_v034_]], [[VAR_v045_]]{{.}} : tensor<1x2x2x2xi8>
// CHECK:             linalg.yield [[VAR_extracted_]] : i8
// CHECK:           } -> tensor<1x2x4x4xi8>
// CHECK:           return [[VAR_v001_]] : tensor<1x2x4x4xi8>
// CHECK:         }
