//===- quantization.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --xten-nn-to-tosa --split-input-file | FileCheck %s

module attributes{} {
// CHECK-LABEL:    func.func @explicit_case
// CHECK-SAME:     ([[PARAM_0_:%.+]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK-DAG:         [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:         [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x3x4x4xf32>}> : () -> tensor<1x3x4x4xf32>
// CHECK-DAG:         [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK:             [[VAR_3_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_2_]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             [[VAR_4_:%.+]] = tosa.add [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             [[VAR_5_:%.+]] = tosa.cast [[VAR_4_]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK:             [[VAR_6_:%.+]] = tosa.cast [[VAR_5_]] : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
// CHECK:             [[VAR_7_:%.+]] = tosa.sub [[VAR_6_]], [[VAR_1_]] : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = tosa.mul [[VAR_7_]], [[VAR_0_]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return [[VAR_8_]] : tensor<1x3x4x4xf32>
// CHECK:           }
    func.func @explicit_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {scale = 3.125000e-02 : f32, zero_point = 1 : i8} -> tensor<1x3x4x4xi8>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xi8>) {scale = 3.125000e-02 : f32, zero_point = 1 : i8} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:    func.func @explicit_case_bf16
// CHECK-SAME:     ([[PARAM_0_:%.+]]: tensor<1x3x4x4xbf16>) -> tensor<1x3x4x4xbf16> {
// CHECK-DAG:         [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xbf16>}> : () -> tensor<1x1x1x1xbf16>
// CHECK-DAG:         [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x3x4x4xbf16>}> : () -> tensor<1x3x4x4xbf16>
// CHECK-DAG:         [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xbf16>}> : () -> tensor<1x1x1x1xbf16>
// CHECK:             [[VAR_3_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_2_]] {shift = 0 : i8} : (tensor<1x3x4x4xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x3x4x4xbf16>
// CHECK:             [[VAR_4_:%.+]] = tosa.add [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x4x4xbf16>, tensor<1x3x4x4xbf16>) -> tensor<1x3x4x4xbf16>
// CHECK:             [[VAR_5_:%.+]] = tosa.cast [[VAR_4_]] : (tensor<1x3x4x4xbf16>) -> tensor<1x3x4x4xi8>
// CHECK:             [[VAR_6_:%.+]] = tosa.cast [[VAR_5_]] : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xbf16>
// CHECK:             [[VAR_7_:%.+]] = tosa.sub [[VAR_6_]], [[VAR_1_]] : (tensor<1x3x4x4xbf16>, tensor<1x3x4x4xbf16>) -> tensor<1x3x4x4xbf16>
// CHECK:             [[VAR_8_:%.+]] = tosa.mul [[VAR_7_]], [[VAR_0_]] {shift = 0 : i8} : (tensor<1x3x4x4xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x3x4x4xbf16>
// CHECK:             return [[VAR_8_]] : tensor<1x3x4x4xbf16>
// CHECK:           }
    func.func @explicit_case_bf16(%arg0: tensor<1x3x4x4xbf16>) -> tensor<1x3x4x4xbf16> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xbf16>) {scale = 3.125000e-02 : f32, zero_point = 1 : i8} -> tensor<1x3x4x4xi8>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xi8>) {scale = 3.125000e-02 : f32, zero_point = 1 : i8} -> tensor<1x3x4x4xbf16>
        return %1 : tensor<1x3x4x4xbf16>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @small_tensors(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32> {
// CHECK-DAG:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK-DAG:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<8.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK-DAG:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
// CHECK-DAG:             %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<2x3xf32>) -> tensor<2x3xi8>
// CHECK-DAG:             %[[VAL_5:.*]] = tosa.clamp %[[VAL_4]] {max_fp = 7.000000e+00 : f32, max_int = 7 : i64, min_fp = -8.000000e+00 : f32, min_int = -8 : i64} : (tensor<2x3xi8>) -> tensor<2x3xi8>
// CHECK-DAG:             %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<2x3xi8>) -> tensor<2x3xf32>
// CHECK-DAG:             %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_2]] {shift = 0 : i8} : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
// CHECK-DAG:             return %[[VAL_7]] : tensor<2x3xf32>
    func.func @small_tensors(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<2x3xf32>) {scale = 8.0 : f32, shift = 3 : si32, zero_point = 0 : i4} -> tensor<2x3xi4>
        %1 = xten_nn.dequantize(%0 : tensor<2x3xi4>) {scale = 8.0 : f32, shift = 3 : si32, zero_point = 0 : i4} -> tensor<2x3xf32>
        return %1 : tensor<2x3xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @quantize_case(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8> {
// CHECK-DAG:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_2:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_3:.*]] = tosa.cast %[[VAL_2]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK-DAG:             return %[[VAL_3]] : tensor<1x3x4x4xi8>
    func.func @quantize_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i8} -> tensor<1x3x4x4xi8>
        return %0 : tensor<1x3x4x4xi8>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @dequantize_case(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32> {
// CHECK-DAG:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_3:.*]] = tosa.cast %[[VAL_0]] : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_4:.*]] = tosa.mul %[[VAL_3]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             return %[[VAL_4]] : tensor<1x3x4x4xf32>
    func.func @dequantize_case(%arg0: tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.dequantize(%arg0 : tensor<1x3x4x4xi8>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i8} -> tensor<1x3x4x4xf32>
        return %0 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @i16_case(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK-DAG:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
// CHECK-DAG:             %[[VAL_5:.*]] = tosa.cast %[[VAL_4]] : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_6:.*]] = tosa.mul %[[VAL_5]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             return %[[VAL_6]] : tensor<1x3x4x4xf32>
    func.func @i16_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i16} -> tensor<1x3x4x4xi16>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xi16>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i16} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @i12_case(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK-DAG:             %[[VAL_1:.*]] = "tosa.const"() <{value = dense<3.200000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_2:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:             %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_1]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
// CHECK-DAG:             %[[VAL_5:.*]] = tosa.clamp %[[VAL_4]] {max_fp = 2.047000e+03 : f32, max_int = 2047 : i64, min_fp = -2.048000e+03 : f32, min_int = -2048 : i64} : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xi16>
// CHECK-DAG:             %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_2]] {shift = 0 : i8} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK-DAG:             return %[[VAL_7]] : tensor<1x3x4x4xf32>
    func.func @i12_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i12} -> tensor<1x3x4x4xi12>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xi12>) {scale = 3.125000e-02 : f32, shift = -5 : si32, zero_point = 0 : i12} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}
