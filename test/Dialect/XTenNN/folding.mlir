// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --test-constant-fold --cse --split-input-file %s -o - | FileCheck %s

func.func @simple_dqq_fold(%arg0: tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %1 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "xten_nn.quantize"(%2) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %4 = "xten_nn.dequantize"(%3) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  return %4 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL: simple_dqq_fold
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize
// CHECK-NOT: xten_nn.quantize
// CHECK-NOT: xten_nn.dequantize

// -----

func.func @no_fold(%arg0: tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %0 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %1 = "xten_nn.dequantize"(%0) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  return %1 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL: no_fold
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize

// -----

func.func @no_dqq_fold_multiple_uses(%arg0: tensor<1x2x7x7xf32>) -> (tensor<1x2x7x7xf32>, tensor<1x2x7x7xi8>) {
  %1 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "xten_nn.quantize"(%2) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  return %2, %3 : tensor<1x2x7x7xf32>, tensor<1x2x7x7xi8>
}

// CHECK-LABEL: no_dqq_fold_multiple_uses
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize
// CHECK: xten_nn.quantize

// -----

func.func @no_dqq_fold_different_type(%arg0: tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %1 = "xten_nn.quantize"(%arg0) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi16>
  %2 = "xten_nn.dequantize"(%1) {shift = -3 : si32} : (tensor<1x2x7x7xi16>) -> tensor<1x2x7x7xf32>
  %3 = "xten_nn.quantize"(%2) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %4 = "xten_nn.dequantize"(%3) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  return %4 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL: no_dqq_fold_different_type
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize

// -----

func.func @no_dqq_fold_different_shift(%arg0: tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32> {
  %1 = "xten_nn.quantize"(%arg0) {shift = -4 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %2 = "xten_nn.dequantize"(%1) {shift = -4 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  %3 = "xten_nn.quantize"(%2) {shift = -3 : si32} : (tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xi8>
  %4 = "xten_nn.dequantize"(%3) {shift = -3 : si32} : (tensor<1x2x7x7xi8>) -> tensor<1x2x7x7xf32>
  return %4 : tensor<1x2x7x7xf32>
}

// CHECK-LABEL: no_dqq_fold_different_shift
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize
// CHECK: xten_nn.quantize
// CHECK: xten_nn.dequantize