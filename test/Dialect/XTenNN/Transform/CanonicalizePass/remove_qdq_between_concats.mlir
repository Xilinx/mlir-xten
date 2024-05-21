// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -xtennn-canonicalize="allow-destructive" -split-input-file | FileCheck %s
// RUN: aten-opt %s -xtennn-canonicalize -split-input-file | FileCheck %s --check-prefix=SANE

func.func @single_qdq(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xf32> {
  %0 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xi8>
  %1 = "xten_nn.dequantize"(%0) {shift = -3 : si32} : (tensor<1x1x7x7xi8>) -> tensor<1x1x7x7xf32>
  return %1 : tensor<1x1x7x7xf32>
}

// CHECK-LABEL:   func.func @single_qdq(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x1x7x7xf32>) {shift = -3 : si32} -> tensor<1x1x7x7xi8>
// CHECK:           %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x1x7x7xi8>) {shift = -3 : si32} -> tensor<1x1x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x1x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @single_qdq(
// SANE-SAME:                          %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x1x7x7xf32>) {shift = -3 : si32} -> tensor<1x1x7x7xi8>
// SANE:           %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x1x7x7xi8>) {shift = -3 : si32} -> tensor<1x1x7x7xf32>
// SANE:           return %[[VAL_2]] : tensor<1x1x7x7xf32>
// SANE:         }

// -----

func.func @single_concat_at_input(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  return %2 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL:   func.func @single_concat_at_input(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// CHECK:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x2x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @single_concat_at_input(
// SANE-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           return %[[VAL_3]] : tensor<1x2x7x7xf32>
// SANE:         }

// -----

func.func @single_concat_at_output(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xi8>
  %1 = "xten_nn.dequantize"(%0) {shift = -3 : si32} : (tensor<1x1x7x7xi8>) -> tensor<1x1x7x7xf32>
  %2 = "tosa.concat"(%1, %1) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  return %2 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL:   func.func @single_concat_at_output(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x1x7x7xf32>) {shift = -3 : si32} -> tensor<1x1x7x7xi8>
// CHECK:           %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x1x7x7xi8>) {shift = -3 : si32} -> tensor<1x1x7x7xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.concat %[[VAL_2]], %[[VAL_2]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           return %[[VAL_3]] : tensor<1x2x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @single_concat_at_output(
// SANE-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x1x7x7xf32>) {shift = -3 : si32} -> tensor<1x1x7x7xi8>
// SANE:           %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x1x7x7xi8>) {shift = -3 : si32} -> tensor<1x1x7x7xf32>
// SANE:           %[[VAL_3:.*]] = tosa.concat %[[VAL_2]], %[[VAL_2]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           return %[[VAL_3]] : tensor<1x2x7x7xf32>
// SANE:         }

// -----

func.func @non_foldable_concats(%arg0: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "tosa.concat"(%2, %2) {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
  return %3 : tensor<2x2x7x7xf32>
}

// CHECK-LABEL:   func.func @non_foldable_concats(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// CHECK:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// CHECK:           return %[[VAL_4]] : tensor<2x2x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @non_foldable_concats(
// SANE-SAME:                                    %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<2x2x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// SANE:           return %[[VAL_4]] : tensor<2x2x7x7xf32>
// SANE:         }

// -----

func.func @foldable_concats(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "tosa.concat"(%2, %2) {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %3 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @foldable_concats(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x4x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @foldable_concats(
// SANE-SAME:                                %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// SANE:           return %[[VAL_4]] : tensor<1x4x7x7xf32>
// SANE:         }

// -----

func.func @multiple_foldable_user_concats(%arg0: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>) {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "tosa.concat"(%2, %2) {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  %4 = "tosa.concat"(%2, %2) {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %3, %4 : tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @multiple_foldable_user_concats(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>) {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @multiple_foldable_user_concats(
// SANE-SAME:                                              %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>) {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// SANE:           %[[VAL_5:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// SANE:           return %[[VAL_4]], %[[VAL_5]] : tensor<1x4x7x7xf32>, tensor<1x4x7x7xf32>
// SANE:         }

// -----

func.func @partially_foldable_user_concats(%arg0: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>) {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "tosa.concat"(%2, %2) {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32> // This is foldable
  %4 = "tosa.concat"(%2, %2) {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32> // This is not foldable
  return %3, %4 : tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>
}

// CHECK-LABEL:   func.func @partially_foldable_user_concats(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>) {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// CHECK:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// CHECK:           return %[[VAL_4]], %[[VAL_5]] : tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @partially_foldable_user_concats(
// SANE-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> (tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>) {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// SANE:           %[[VAL_5:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 0 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<2x2x7x7xf32>
// SANE:           return %[[VAL_4]], %[[VAL_5]] : tensor<1x4x7x7xf32>, tensor<2x2x7x7xf32>
// SANE:         }

// -----

func.func @qdq_different_shift(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "xten_nn.quantize"(%0) {shift = -5 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "tosa.concat"(%2, %2) {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %3 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @qdq_different_shift(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -5 : si32} -> tensor<1x2x7x7xi8>
// CHECK:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x4x7x7xf32>
// CHECK:         }

// SANE-LABEL:   func.func @qdq_different_shift(
// SANE-SAME:                                   %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// SANE:           %[[VAL_1:.*]] = tosa.concat %[[VAL_0]], %[[VAL_0]] {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_1]] : tensor<1x2x7x7xf32>) {shift = -5 : si32} -> tensor<1x2x7x7xi8>
// SANE:           %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x2x7x7xi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
// SANE:           %[[VAL_4:.*]] = tosa.concat %[[VAL_3]], %[[VAL_3]] {axis = 1 : i32} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// SANE:           return %[[VAL_4]] : tensor<1x4x7x7xf32>
// SANE:         }
