// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-linalg -split-input-file %s | FileCheck %s

func.func @grid_sample_bilinear_zeros(%arg0: tensor<1x2x4x4xf32>, %arg1: tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32> {
  %0 = xten_nn.grid_sample %arg0, %arg1 {align_corners = 1 : i64, mode = 0 : i64, padding_mode = 0 : i64} : (tensor<1x2x4x4xf32>, tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @grid_sample_bilinear_zeros
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x2x4x4xf32>, [[GRID:%.+]]: tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x2x3x3xf32>
// CHECK:         [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[EMPTY]] : tensor<1x2x3x3xf32>) {
// CHECK:         ^bb0({{%.+}}: f32):
// CHECK-DAG:       [[N:%.+]] = linalg.index 0 : index
// CHECK-DAG:       [[C:%.+]] = linalg.index 1 : index
// CHECK-DAG:       [[OH:%.+]] = linalg.index 2 : index
// CHECK-DAG:       [[OW:%.+]] = linalg.index 3 : index
// CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[GRID_X:%.+]] = tensor.extract [[GRID]]{{\[}}[[N]], [[OH]], [[OW]], [[C0]]] : tensor<1x3x3x2xf32>
// CHECK-DAG:       [[GRID_Y:%.+]] = tensor.extract [[GRID]]{{\[}}[[N]], [[OH]], [[OW]], [[C1]]] : tensor<1x3x3x2xf32>
// CHECK-DAG:       [[F1:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[FHALF:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:           [[X_SHIFTED:%.+]] = arith.addf [[GRID_X]], [[F1]] : f32
// CHECK:           [[F3:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:           [[X_SCALED:%.+]] = arith.mulf [[X_SHIFTED]], [[F3]] : f32
// CHECK:           [[X:%.+]] = arith.mulf [[X_SCALED]], [[FHALF]] : f32
// CHECK-DAG:       [[F1_Y:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[FHALF_Y:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:           [[Y_SHIFTED:%.+]] = arith.addf [[GRID_Y]], [[F1_Y]] : f32
// CHECK:           [[F3_Y:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:           [[Y_SCALED:%.+]] = arith.mulf [[Y_SHIFTED]], [[F3_Y]] : f32
// CHECK:           [[Y:%.+]] = arith.mulf [[Y_SCALED]], [[FHALF_Y]] : f32
// CHECK-DAG:       [[X0F:%.+]] = math.floor [[X]] : f32
// CHECK-DAG:       [[Y0F:%.+]] = math.floor [[Y]] : f32
// CHECK-DAG:       [[X0I64:%.+]] = arith.fptosi [[X0F]] : f32 to i64
// CHECK-DAG:       [[X0:%.+]] = arith.index_cast [[X0I64]] : i64 to index
// CHECK-DAG:       [[Y0I64:%.+]] = arith.fptosi [[Y0F]] : f32 to i64
// CHECK-DAG:       [[Y0:%.+]] = arith.index_cast [[Y0I64]] : i64 to index
// CHECK-DAG:       [[X1:%.+]] = arith.addi [[X0]], [[C1]] : index
// CHECK-DAG:       [[Y1:%.+]] = arith.addi [[Y0]], [[C1]] : index
// CHECK-DAG:       [[X_LERP:%.+]] = arith.subf [[X]], [[X0F]] : f32
// CHECK-DAG:       [[Y_LERP:%.+]] = arith.subf [[Y]], [[Y0F]] : f32
// CHECK-DAG:       [[F1_W:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[X0_WEIGHT:%.+]] = arith.subf [[F1_W]], [[X_LERP]] : f32
// CHECK-DAG:       [[Y0_WEIGHT:%.+]] = arith.subf [[F1_W]], [[Y_LERP]] : f32
// CHECK:           [[Y0_IN_BOUNDS:%.+]] = arith.andi {{%.+}}, {{%.+}} : i1
// CHECK:           [[X0_IN_BOUNDS:%.+]] = arith.andi {{%.+}}, {{%.+}} : i1
// CHECK:           [[Y0_X0_IN_BOUNDS:%.+]] = arith.andi [[Y0_IN_BOUNDS]], [[X0_IN_BOUNDS]] : i1
// CHECK:           [[Y0_CLAMP_LOW:%.+]] = arith.select {{%.+}}, {{%.+}}, [[Y0]] : index
// CHECK:           [[Y0_CLAMPED:%.+]] = arith.select {{%.+}}, {{%.+}}, [[Y0_CLAMP_LOW]] : index
// CHECK:           [[X0_CLAMP_LOW:%.+]] = arith.select {{%.+}}, {{%.+}}, [[X0]] : index
// CHECK:           [[X0_CLAMPED:%.+]] = arith.select {{%.+}}, {{%.+}}, [[X0_CLAMP_LOW]] : index
// CHECK-DAG:       [[SAMPLE_Y0_X0:%.+]] = tensor.extract [[INPUT]]{{\[}}[[N]], [[C]], [[Y0_CLAMPED]], [[X0_CLAMPED]]] : tensor<1x2x4x4xf32>
// CHECK-DAG:       [[FZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[PADDED_Y0_X0:%.+]] = arith.select [[Y0_X0_IN_BOUNDS]], [[SAMPLE_Y0_X0]], [[FZERO]] : f32
// CHECK-DAG:       [[WEIGHT_Y0_X0:%.+]] = arith.mulf [[Y0_WEIGHT]], [[X0_WEIGHT]] : f32
// CHECK:           [[ACC0:%.+]] = arith.mulf [[PADDED_Y0_X0]], [[WEIGHT_Y0_X0]] : f32
// CHECK:           [[SAMPLE_Y0_X1:%.+]] = tensor.extract [[INPUT]]{{\[}}[[N]], [[C]], {{%.+}}, {{%.+}}] : tensor<1x2x4x4xf32>
// CHECK:           [[PADDED_Y0_X1:%.+]] = arith.select {{%.+}}, [[SAMPLE_Y0_X1]], {{%.+}} : f32
// CHECK:           [[WEIGHT_Y0_X1:%.+]] = arith.mulf [[Y0_WEIGHT]], [[X_LERP]] : f32
// CHECK:           [[TERM_Y0_X1:%.+]] = arith.mulf [[PADDED_Y0_X1]], [[WEIGHT_Y0_X1]] : f32
// CHECK:           [[ACC1:%.+]] = arith.addf [[ACC0]], [[TERM_Y0_X1]] : f32
// CHECK:           [[SAMPLE_Y1_X0:%.+]] = tensor.extract [[INPUT]]{{\[}}[[N]], [[C]], {{%.+}}, {{%.+}}] : tensor<1x2x4x4xf32>
// CHECK:           [[PADDED_Y1_X0:%.+]] = arith.select {{%.+}}, [[SAMPLE_Y1_X0]], {{%.+}} : f32
// CHECK:           [[WEIGHT_Y1_X0:%.+]] = arith.mulf [[Y_LERP]], [[X0_WEIGHT]] : f32
// CHECK:           [[TERM_Y1_X0:%.+]] = arith.mulf [[PADDED_Y1_X0]], [[WEIGHT_Y1_X0]] : f32
// CHECK:           [[ACC2:%.+]] = arith.addf [[ACC1]], [[TERM_Y1_X0]] : f32
// CHECK:           [[SAMPLE_Y1_X1:%.+]] = tensor.extract [[INPUT]]{{\[}}[[N]], [[C]], {{%.+}}, {{%.+}}] : tensor<1x2x4x4xf32>
// CHECK:           [[PADDED_Y1_X1:%.+]] = arith.select {{%.+}}, [[SAMPLE_Y1_X1]], {{%.+}} : f32
// CHECK:           [[WEIGHT_Y1_X1:%.+]] = arith.mulf [[Y_LERP]], [[X_LERP]] : f32
// CHECK:           [[TERM_Y1_X1:%.+]] = arith.mulf [[PADDED_Y1_X1]], [[WEIGHT_Y1_X1]] : f32
// CHECK:           [[ACC3:%.+]] = arith.addf [[ACC2]], [[TERM_Y1_X1]] : f32
// CHECK:           linalg.yield [[ACC3]] : f32
// CHECK:         } -> tensor<1x2x3x3xf32>
// CHECK:         return [[GENERIC]] : tensor<1x2x3x3xf32>
// CHECK-NOT: xten_nn.grid_sample

// -----

func.func @grid_sample_nearest_border_bf16(%arg0: tensor<1x2x4x4xbf16>, %arg1: tensor<1x3x3x2xbf16>) -> tensor<1x2x3x3xbf16> {
  %0 = xten_nn.grid_sample %arg0, %arg1 {align_corners = 0 : i64, mode = 1 : i64, padding_mode = 1 : i64} : (tensor<1x2x4x4xbf16>, tensor<1x3x3x2xbf16>) -> tensor<1x2x3x3xbf16>
  return %0 : tensor<1x2x3x3xbf16>
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @grid_sample_nearest_border_bf16
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x2x4x4xbf16>, [[GRID:%.+]]: tensor<1x3x3x2xbf16>) -> tensor<1x2x3x3xbf16>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x2x3x3xbf16>
// CHECK:         [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs([[EMPTY]] : tensor<1x2x3x3xbf16>) {
// CHECK:         ^bb0({{%.+}}: bf16):
// CHECK-DAG:       [[N:%.+]] = linalg.index 0 : index
// CHECK-DAG:       [[C:%.+]] = linalg.index 1 : index
// CHECK-DAG:       [[OH:%.+]] = linalg.index 2 : index
// CHECK-DAG:       [[OW:%.+]] = linalg.index 3 : index
// CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[GRID_X_BF16:%.+]] = tensor.extract [[GRID]]{{\[}}[[N]], [[OH]], [[OW]], [[C0]]] : tensor<1x3x3x2xbf16>
// CHECK-DAG:       [[GRID_Y_BF16:%.+]] = tensor.extract [[GRID]]{{\[}}[[N]], [[OH]], [[OW]], [[C1]]] : tensor<1x3x3x2xbf16>
// CHECK:           [[GRID_X:%.+]] = arith.extf [[GRID_X_BF16]] : bf16 to f32
// CHECK-DAG:       [[F1:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[FHALF:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:           [[X_SHIFTED:%.+]] = arith.addf [[GRID_X]], [[F1]] : f32
// CHECK:           [[F4:%.+]] = arith.constant 4.000000e+00 : f32
// CHECK:           [[X_SCALED:%.+]] = arith.mulf [[X_SHIFTED]], [[F4]] : f32
// CHECK:           [[X_CENTERED:%.+]] = arith.subf [[X_SCALED]], [[F1]] : f32
// CHECK:           [[X:%.+]] = arith.mulf [[X_CENTERED]], [[FHALF]] : f32
// CHECK:           [[GRID_Y:%.+]] = arith.extf [[GRID_Y_BF16]] : bf16 to f32
// CHECK-DAG:       [[F1_Y:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[FHALF_Y:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK:           [[Y_SHIFTED:%.+]] = arith.addf [[GRID_Y]], [[F1_Y]] : f32
// CHECK:           [[F4_Y:%.+]] = arith.constant 4.000000e+00 : f32
// CHECK:           [[Y_SCALED:%.+]] = arith.mulf [[Y_SHIFTED]], [[F4_Y]] : f32
// CHECK:           [[Y_CENTERED:%.+]] = arith.subf [[Y_SCALED]], [[F1_Y]] : f32
// CHECK:           [[Y:%.+]] = arith.mulf [[Y_CENTERED]], [[FHALF_Y]] : f32
// CHECK-DAG:       [[X_FLOOR:%.+]] = math.floor [[X]] : f32
// CHECK:           [[X_I64:%.+]] = arith.fptosi [[X_FLOOR]] : f32 to i64
// CHECK-DAG:       [[X0:%.+]] = arith.index_cast [[X_I64]] : i64 to index
// CHECK-DAG:       [[X1:%.+]] = arith.addi [[X0]], {{%.+}} : index
// CHECK-DAG:       [[X_FRAC:%.+]] = arith.subf [[X]], [[X_FLOOR]] : f32
// CHECK-DAG:       [[X_GT_HALF:%.+]] = arith.cmpf ogt, [[X_FRAC]], {{%.+}} : f32
// CHECK-DAG:       [[X_EQ_HALF:%.+]] = arith.cmpf oeq, [[X_FRAC]], {{%.+}} : f32
// CHECK-DAG:       [[X0_I64:%.+]] = arith.index_cast [[X0]] : index to i64
// CHECK-DAG:       [[X_REM:%.+]] = arith.remsi [[X0_I64]], {{%.+}} : i64
// CHECK:           [[X_ODD:%.+]] = arith.cmpi ne, [[X_REM]], {{%.+}} : i64
// CHECK:           [[X_TIE_UP:%.+]] = arith.andi [[X_EQ_HALF]], [[X_ODD]] : i1
// CHECK:           [[X_USE_UP:%.+]] = arith.ori [[X_GT_HALF]], [[X_TIE_UP]] : i1
// CHECK-DAG:       [[X_NEAREST:%.+]] = arith.select [[X_USE_UP]], [[X1]], [[X0]] : index
// CHECK-DAG:       [[Y_FLOOR:%.+]] = math.floor [[Y]] : f32
// CHECK:           [[Y_I64:%.+]] = arith.fptosi [[Y_FLOOR]] : f32 to i64
// CHECK-DAG:       [[Y0:%.+]] = arith.index_cast [[Y_I64]] : i64 to index
// CHECK-DAG:       [[Y1:%.+]] = arith.addi [[Y0]], {{%.+}} : index
// CHECK-DAG:       [[Y_FRAC:%.+]] = arith.subf [[Y]], [[Y_FLOOR]] : f32
// CHECK-DAG:       [[Y_GT_HALF:%.+]] = arith.cmpf ogt, [[Y_FRAC]], {{%.+}} : f32
// CHECK-DAG:       [[Y_EQ_HALF:%.+]] = arith.cmpf oeq, [[Y_FRAC]], {{%.+}} : f32
// CHECK-DAG:       [[Y0_I64:%.+]] = arith.index_cast [[Y0]] : index to i64
// CHECK-DAG:       [[Y_REM:%.+]] = arith.remsi [[Y0_I64]], {{%.+}} : i64
// CHECK:           [[Y_ODD:%.+]] = arith.cmpi ne, [[Y_REM]], {{%.+}} : i64
// CHECK:           [[Y_TIE_UP:%.+]] = arith.andi [[Y_EQ_HALF]], [[Y_ODD]] : i1
// CHECK:           [[Y_USE_UP:%.+]] = arith.ori [[Y_GT_HALF]], [[Y_TIE_UP]] : i1
// CHECK-DAG:       [[Y_NEAREST:%.+]] = arith.select [[Y_USE_UP]], [[Y1]], [[Y0]] : index
// CHECK:           [[Y_CLAMP_LOW:%.+]] = arith.select {{%.+}}, {{%.+}}, [[Y_NEAREST]] : index
// CHECK:           [[Y_CLAMPED:%.+]] = arith.select {{%.+}}, {{%.+}}, [[Y_CLAMP_LOW]] : index
// CHECK:           [[X_CLAMP_LOW:%.+]] = arith.select {{%.+}}, {{%.+}}, [[X_NEAREST]] : index
// CHECK:           [[X_CLAMPED:%.+]] = arith.select {{%.+}}, {{%.+}}, [[X_CLAMP_LOW]] : index
// CHECK:           [[SAMPLE:%.+]] = tensor.extract [[INPUT]]{{\[}}[[N]], [[C]], [[Y_CLAMPED]], [[X_CLAMPED]]] : tensor<1x2x4x4xbf16>
// CHECK:           linalg.yield [[SAMPLE]] : bf16
// CHECK:         } -> tensor<1x2x3x3xbf16>
// CHECK:         return [[GENERIC]] : tensor<1x2x3x3xbf16>
// CHECK-NOT: xten_nn.grid_sample
