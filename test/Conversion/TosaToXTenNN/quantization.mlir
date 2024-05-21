//===- quantization.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --tosa-to-xten-nn --split-input-file | FileCheck %s
module attributes {} {
// CHECK-LABEL:     func.func @explicit_case(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @explicit_case(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %18 = "tosa.const"() {value = dense<0> : tensor<1x1x1x1xi8>} : () -> tensor<1x1x1x1xi8>
    %19 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x3x4x4xf32>} : () -> tensor<1x3x4x4xf32>
    %20 = "tosa.reciprocal"(%19) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %21 = "tosa.mul"(%arg0, %20) { shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %22 = "tosa.cast"(%21) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %23 = "tosa.add"(%22, %18) {} : (tensor<1x3x4x4xi8>, tensor<1x1x1x1xi8>) -> tensor<1x3x4x4xi8>
    %24 = "tosa.sub"(%23, %18) {} : (tensor<1x3x4x4xi8>, tensor<1x1x1x1xi8>) -> tensor<1x3x4x4xi8>
    %25 = "tosa.cast"(%24) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %26 = "tosa.mul"(%25, %19) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %26 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @recip_folded(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @recip_folded(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %17 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %18 = "tosa.const"() {value = dense<0> : tensor<1x1x1x1xi8>} : () -> tensor<1x1x1x1xi8>
    %19 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %21 = "tosa.mul"(%arg0, %17) { shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %22 = "tosa.cast"(%21) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %23 = "tosa.add"(%22, %18) {} : (tensor<1x3x4x4xi8>, tensor<1x1x1x1xi8>) -> tensor<1x3x4x4xi8>
    %24 = "tosa.sub"(%23, %18) {} : (tensor<1x3x4x4xi8>, tensor<1x1x1x1xi8>) -> tensor<1x3x4x4xi8>
    %25 = "tosa.cast"(%24) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %26 = "tosa.mul"(%25, %19) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %26 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @recip_add_folded(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @recip_add_folded(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %17 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %18 = "tosa.const"() {value = dense<0> : tensor<1x1x1x1xi8>} : () -> tensor<1x1x1x1xi8>
    %19 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %21 = "tosa.mul"(%arg0, %17) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %22 = "tosa.cast"(%21) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %24 = "tosa.sub"(%22, %18) {} : (tensor<1x3x4x4xi8>, tensor<1x1x1x1xi8>) -> tensor<1x3x4x4xi8>
    %25 = "tosa.cast"(%24) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %26 = "tosa.mul"(%25, %19) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %26 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @all_ops_folded(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @all_ops_folded(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %17 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %19 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %21 = "tosa.mul"(%arg0, %17) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %22 = "tosa.cast"(%21) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %25 = "tosa.cast"(%22) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %26 = "tosa.mul"(%25, %19) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %26 : tensor<1x3x4x4xf32>
  }
}


// --

module attributes {} {
// CHECK-LABEL:     func.func @missing_dq_cast_mul(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_3:.*]] = tosa.cast %[[VAL_2]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK:             return %[[VAL_3]] : tensor<1x3x4x4xi8>
// CHECK:           }
  func.func @missing_dq_cast_mul(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xi8> {
    %17 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %19 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %21 = "tosa.mul"(%arg0, %17) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %22 = "tosa.cast"(%21) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    return %22 : tensor<1x3x4x4xi8>
  }
}

// --

// For the implementation of i16 in the future we would expect a pattern:
//   f32->cast->i32->clip->i32->cast->i16->cast->f32
// but for now we do nothing.
module attributes {} {
// CHECK-LABEL:     func.func @i16_quantization(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = tosa.cast %[[VAL_0]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
// CHECK:             %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @i16_quantization(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.cast"(%arg0) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
    %1 = "tosa.cast"(%0) : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
    return %1 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @mul_missing_dequantize(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @mul_missing_dequantize(%arg0: tensor<1x3x4x4xf32> ) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %1 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @mul_missing_quantize(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_0]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_3:.*]] = tosa.mul %[[VAL_2]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_3]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @mul_missing_quantize(%arg0: tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = xten_nn.dequantize(%arg0 : tensor<1x3x4x4xi8>) {shift = 0 : si32}  -> tensor<1x3x4x4xf32>
    %2 = "tosa.mul"(%1, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %2 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @unequal_mul_constants(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.000000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = xten_nn.quantize(%[[VAL_3]] : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_5:.*]] = xten_nn.dequantize(%[[VAL_4]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_6:.*]] = tosa.mul %[[VAL_5]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_6]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @unequal_mul_constants(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.00000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = xten_nn.quantize(%2 : tensor<1x3x4x4xf32>) {shift = 0 : si32}  -> tensor<1x3x4x4xi8>
    %4 = xten_nn.dequantize(%3 : tensor<1x3x4x4xi8>) {shift = 0 : si32}  -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %5 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @equal_mul_constants_not_log2(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.000000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_3:.*]] = xten_nn.quantize(%[[VAL_2]] : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_4:.*]] = xten_nn.dequantize(%[[VAL_3]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_5:.*]] = tosa.mul %[[VAL_4]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_5]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @equal_mul_constants_not_log2(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.00000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = xten_nn.quantize(%2 : tensor<1x3x4x4xf32>) {shift = 0 : si32}  -> tensor<1x3x4x4xi8>
    %4 = xten_nn.dequantize(%3 : tensor<1x3x4x4xi8>) {shift = 0 : si32}  -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %5 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @sum_shifts(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -2 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -2 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @sum_shifts(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = xten_nn.quantize(%2 : tensor<1x3x4x4xf32>) {shift = 3 : si32}  -> tensor<1x3x4x4xi8>
    %4 = xten_nn.dequantize(%3 : tensor<1x3x4x4xi8>) {shift = 3 : si32}  -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %5 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @multiple_q_uses(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x3x4x4xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<1x3x4x4xi8>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi8>) {
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_4:.*]] = tosa.mul %[[VAL_0]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_5:.*]] = tosa.cast %[[VAL_4]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_3]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_8:.*]] = tosa.add %[[VAL_1]], %[[VAL_5]] : (tensor<1x3x4x4xi8>, tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xi8>
// CHECK:             return %[[VAL_7]], %[[VAL_8]] : tensor<1x3x4x4xf32>, tensor<1x3x4x4xi8>
// CHECK:           }
  func.func @multiple_q_uses(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x3x4x4xi8>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xi8>) {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    // quantized output is used twice, so we cannot replace the casts here we want strictly Q->DQ
    %6 = "tosa.add"(%arg1, %3) : (tensor<1x3x4x4xi8>, tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xi8>
    return %5,  %6: tensor<1x3x4x4xf32>, tensor<1x3x4x4xi8>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @multiple_dq_uses(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x3x4x4xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_4:.*]] = tosa.mul %[[VAL_0]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_5:.*]] = xten_nn.quantize(%[[VAL_4]] : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_6:.*]] = xten_nn.dequantize(%[[VAL_5]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_3]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_8:.*]] = tosa.add %[[VAL_1]], %[[VAL_6]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_7]], %[[VAL_8]] : tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @multiple_dq_uses(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    // dequantize output is used twice, meaning the muls will not fold.
    %6 = "tosa.add"(%arg1, %4) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %5,  %6: tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @multiple_dq_uses_muls_fold(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x3x4x4xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
// CHECK:             %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = tosa.add %[[VAL_1]], %[[VAL_3]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_3]], %[[VAL_4]] : tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @multiple_dq_uses_muls_fold(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
    %0 = "tosa.const"() {value = dense<1.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<1.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    // dequantize output is used twice, but the scale factor is one so the muls fold away
    %6 = "tosa.add"(%arg1, %4) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %5,  %6: tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @multiple_mul_q_uses(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<1x3x4x4xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_4:.*]] = tosa.mul %[[VAL_0]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_5:.*]] = xten_nn.quantize(%[[VAL_4]] : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_6:.*]] = xten_nn.dequantize(%[[VAL_5]] : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_3]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_8:.*]] = tosa.add %[[VAL_1]], %[[VAL_4]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_7]], %[[VAL_8]] : tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @multiple_mul_q_uses(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    // quantize mul output is used twice this is not allowed, it should only be used by the QuantizeOp
    %6 = "tosa.add"(%arg1, %2) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %5,  %6: tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @multiple_mul_dq_uses(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x3x4x4xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
// CHECK:             %[[VAL_2:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_3:.*]] = xten_nn.dequantize(%[[VAL_2]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = tosa.add %[[VAL_1]], %[[VAL_3]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_3]], %[[VAL_4]] : tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @multiple_mul_dq_uses(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    // dequantize mul output is used twice, but that can be expected.
    %6 = "tosa.add"(%arg1, %5) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %5,  %6: tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @broadcast_mul_on_quantize(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3xf32>) -> tensor<4x3xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK:             %[[VAL_4:.*]] = xten_nn.quantize(%[[VAL_3]] : tensor<4x3xf32>) {shift = 0 : si32} -> tensor<4x3xi8>
// CHECK:             %[[VAL_5:.*]] = xten_nn.dequantize(%[[VAL_4]] : tensor<4x3xi8>) {shift = 0 : si32} -> tensor<4x3xf32>
// CHECK:             %[[VAL_6:.*]] = tosa.mul %[[VAL_5]], %[[VAL_2]] {shift = 0 : i8} : (tensor<4x3xf32>, tensor<1x1xf32>) -> tensor<4x3xf32>
// CHECK:             return %[[VAL_6]] : tensor<4x3xf32>
// CHECK:           }
  func.func @broadcast_mul_on_quantize(%arg0: tensor<1x3xf32>) -> tensor<4x3xf32> {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<4x3xf32>} : () -> tensor<4x3xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
    // Mul cannot be folded because the output shape changes w.r.t input due to broadcasting.
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    %3 = "tosa.cast"(%2) : (tensor<4x3xf32>) -> tensor<4x3xi8>
    %4 = "tosa.cast"(%3) : (tensor<4x3xi8>) -> tensor<4x3xf32>
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<4x3xf32>, tensor<1x1xf32>) -> tensor<4x3xf32>
    return %5 : tensor<4x3xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @broadcast_mul_on_dequantize(
// CHECK-SAME:                                             %[[VAL_0:.*]]: tensor<1x3xf32>) -> tensor<4x3xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
// CHECK:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
// CHECK:             %[[VAL_4:.*]] = xten_nn.quantize(%[[VAL_3]] : tensor<1x3xf32>) {shift = 0 : si32} -> tensor<1x3xi8>
// CHECK:             %[[VAL_5:.*]] = xten_nn.dequantize(%[[VAL_4]] : tensor<1x3xi8>) {shift = 0 : si32} -> tensor<1x3xf32>
// CHECK:             %[[VAL_6:.*]] = tosa.mul %[[VAL_5]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK:             return %[[VAL_6]] : tensor<4x3xf32>
// CHECK:           }
  func.func @broadcast_mul_on_dequantize(%arg0: tensor<1x3xf32>) -> tensor<4x3xf32> {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<4x3xf32>} : () -> tensor<4x3xf32>
    %2 = "tosa.mul"(%arg0, %0) {shift = 0 : i8} : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
    %3 = "tosa.cast"(%2) : (tensor<1x3xf32>) -> tensor<1x3xi8>
    %4 = "tosa.cast"(%3) : (tensor<1x3xi8>) -> tensor<1x3xf32>
    // Mul cannot be folded because the output shape changes w.r.t input due to broadcasting.
    %5 = "tosa.mul"(%4, %1) {shift = 0 : i8} : (tensor<1x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    return %5 : tensor<4x3xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @sort_mul_operands(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
  func.func @sort_mul_operands(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    // The MULs have the constants on operand(0) the SortCommutativeOperands pass should move them to
    // operand(1) and the MUL folding should occur.
    %2 = "tosa.mul"(%0, %arg0) {shift = 0 : i8} : (tensor<1x1x1x1xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %3 = xten_nn.quantize(%2 : tensor<1x3x4x4xf32>) {shift = 0 : si32}  -> tensor<1x3x4x4xi8>
    %4 = xten_nn.dequantize(%3 : tensor<1x3x4x4xi8>) {shift = 0 : si32}  -> tensor<1x3x4x4xf32>
    %5 = "tosa.mul"(%1, %4) {shift = 0 : i8} : (tensor<1x1x1x1xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %5 : tensor<1x3x4x4xf32>
  }
}

// --

module attributes {} {
// CHECK-LABEL:     func.func @fold_after_sort_on_mul() -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_0:.*]] = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x3x4x4xf32>}> : () -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_1:.*]] = xten_nn.quantize(%[[VAL_0]] : tensor<1x3x4x4xf32>) {shift = -7 : si32} -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_2:.*]] = xten_nn.dequantize(%[[VAL_1]] : tensor<1x3x4x4xi8>) {shift = -7 : si32} -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_2]] : tensor<1x3x4x4xf32>
// CHECK:           }
 func.func @fold_after_sort_on_mul() -> tensor<1x3x4x4xf32> {
    %0 = "tosa.const"() {value = dense<1.280000e+02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.const"() {value = dense<7.812500e-03> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %2 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x3x4x4xf32>} : () -> tensor<1x3x4x4xf32>
    %3 = "tosa.mul"(%0, %2) {shift = 0 : i8} : (tensor<1x1x1x1xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %4 = xten_nn.quantize(%3 : tensor<1x3x4x4xf32>) {shift = 0 : si32} -> tensor<1x3x4x4xi8>
    %5 = xten_nn.dequantize(%4 : tensor<1x3x4x4xi8>) {shift = 0 : si32} -> tensor<1x3x4x4xf32>
    %6 = "tosa.mul"(%5, %1) {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    return %6 : tensor<1x3x4x4xf32>
  }
}
