// RUN: aten-opt %s -xten-canonicalize -split-input-file | FileCheck %s

func.func @single_concat(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  return %0 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL:   func.func @single_concat(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x2x7x7xf32>
// CHECK:         }

// -----

func.func @simple_fold(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tensor.empty() : tensor<1x2x7x7xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %2 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @simple_fold(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_1]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @concat_different_axis(%arg0: tensor<1x1x7x7xf32>) -> tensor<2x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 0 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<2x1x7x7xf32>
  %1 = tensor.empty() : tensor<2x3x7x7xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1 : i64} : (tensor<2x1x7x7xf32>, tensor<2x3x7x7xf32>) -> tensor<2x4x7x7xf32>
  return %2 : tensor<2x4x7x7xf32>
}

// CHECK-LABEL:   func.func @concat_different_axis(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<2x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]]) {axis = 0 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<2x1x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x3x7x7xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_2]]) {axis = 1 : i64} : (tensor<2x1x7x7xf32>, tensor<2x3x7x7xf32>) -> tensor<2x4x7x7xf32>
// CHECK:           return %[[VAL_3]] : tensor<2x4x7x7xf32>
// CHECK:         }

// -----

func.func @fold_repeated_operand(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%0, %0) {axis = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %1 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @fold_repeated_operand(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_0]], %[[VAL_0]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_1]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @nested_fold(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x6x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tensor.empty() : tensor<1x2x7x7xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  %3 = "tosa.concat"(%2, %1) {axis = 1 : i64} : (tensor<1x4x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x6x7x7xf32>
  return %3 : tensor<1x6x7x7xf32>
}

// CHECK-LABEL:   func.func @nested_fold(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x6x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x6x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x6x7x7xf32>
// CHECK:         }

// -----

func.func @wide_fold(%arg0: tensor<1x1x7x7xf32>, %arg1: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = "tosa.concat"(%arg1, %arg1) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %2 = "tosa.concat"(%0, %1) {axis = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %2 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @wide_fold(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<1x1x7x7xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }

// -----

func.func @fold_with_qdq(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %0 = "tosa.concat"(%arg0, %arg0) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = xten_nn.quantize (%0: tensor<1x2x7x7xf32>) {shift = -3 : si32} -> tensor<1x2x7x7xsi8>
  %2 = xten_nn.dequantize (%1: tensor<1x2x7x7xsi8>) {shift = -3 : si32} -> tensor<1x2x7x7xf32>
  %3 = tensor.empty() : tensor<1x2x7x7xf32>
  %4 = "tosa.concat"(%2, %3) {axis = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %4 : tensor<1x4x7x7xf32>
}

// CHECK-LABEL:   func.func @fold_with_qdq(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x2x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_0]], %[[VAL_0]], %[[VAL_1]]) {axis = 1 : i64} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>) -> tensor<1x4x7x7xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x7x7xf32>
// CHECK:         }
