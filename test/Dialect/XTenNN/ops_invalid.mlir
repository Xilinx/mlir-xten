// (c) Copyright 2024-2026 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics


func.func @atan2_int(%arg0: tensor<1x10xi4>, %arg1: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.atan2 %arg0, %arg1 : (tensor<1x10xi4>, tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @elu_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.elu %arg0 { alpha = 1.000000e-00 : f32} : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @mish_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.mish %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @kernel_missing_parenthesis() {
    // expected-error@+1 {{expected '('}}
    %a = xten_nn.kernel "myKernel" -> tensor<2xi64>
}

// -----

func.func @kernel_missing_colon(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{expected ':`, (argument format is val : type)}}
    %a = xten_nn.kernel "myKernel" (%arg0, %arg1) -> tensor<2xi64>
}

// -----

func.func @kernel_missing_type(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{expected non-function type}}
    %a = xten_nn.kernel "myKernel" (%arg0 : ) -> tensor<2xi64>
}

// -----

func.func @kernel_trailing_comma(%arg0: i8) {
    // expected-error@+1 {{expected SSA operand}}
    %a = xten_nn.kernel "myKernel" (%arg0 :i8, ) -> tensor<2xi64>
}

// -----

func.func @kernel_missing_name() {
    // expected-error@+1 {{'xten_nn.kernel' invalid kind of attribute specified}}
    %b = xten_nn.kernel () -> tensor<2xi64>
    return
}

// -----

func.func @kernel_missing_result(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{expected non-function type}}
    xten_nn.kernel "myKernel" () ->
}

// -----

func.func @kernel_instantiation_list_different_length(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{instantiation arg names must be either empty or as long as instantiation args}}
    %x = xten_nn.kernel "myKernel" () instantiation_args ["N" = 42 : i32, 51 : index] -> i32
    return
}

// -----

func.func @kernel_instantiation_list_non_empty(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{cannot have instantiation arg names without instantiation args}}
    %x = xten_nn.kernel "myKernel" () {instantiation_arg_names = ["N"]} -> i32
    return
}

// -----

func.func @kernel_instantiation_list_non_quoted(%arg0: i8, %arg1: i8) {
    // expected-error@+1 {{expected ']'}}
    %x = xten_nn.kernel "myKernel" () instantiation_args [N = 42 : i32] -> i32
    return
}


// -----

func.func @topk_wrong_output_shape(%arg0: tensor<10x10xf32>) {
    %k = arith.constant 7 : i64
    // expected-error@+2 {{failed to infer returned types}}
    // expected-error@+1 {{'xten_nn.topk' op inferred type(s) 'tensor<7x10xf32>', 'tensor<7x10xi64>' are incompatible with return type(s) of operation 'tensor<1xf32>', 'tensor<1xi64>'}}
    %a, %b = xten_nn.topk(%arg0 : tensor<10x10xf32>, %k : i64) {axis = 0 : i64, largest = true, sorted = true} -> tensor<1xf32>, tensor<1xi64>
    return
}

// -----

func.func @topk_wrong_indices_shape(%arg0: tensor<10x10xf32>) {
    %k = arith.constant 7 : i64
    // expected-error@+2 {{failed to infer returned types}}
    // expected-error@+1 {{'xten_nn.topk' op inferred type(s) 'tensor<7x10xf32>', 'tensor<7x10xi64>' are incompatible with return type(s) of operation 'tensor<7x10xf32>', 'tensor<7x10xf32>'}}
    %a, %b = xten_nn.topk(%arg0 : tensor<10x10xf32>, %k : i64) {axis = 0 : i64, largest = true, sorted = true} -> tensor<7x10xf32>, tensor<7x10xf32>
    return
}

// -----

func.func @topk_wrong_axis(%arg0: tensor<10x10xf32>) {
    %k = arith.constant 7 : i64
    // expected-error@+2 {{failed to infer returned types}}
    // expected-error@+1 {{expected axis to be within [-rank,rank) (where rank is the rank of the input}}
    %a, %b = xten_nn.topk(%arg0 : tensor<10x10xf32>, %k : i64) {axis = 3 : i64, largest = true, sorted = true} -> tensor<10x10xf32>, tensor<1xi64>
    return
}

// -----

func.func @topk_large_k(%arg0: tensor<10x10xf32>) {
    %k = arith.constant 100 : i64
    // expected-error@+2 {{failed to infer returned types}}
    // expected-error@+1 {{expected k <= dimension size}}
    %a, %b = xten_nn.topk(%arg0 : tensor<10x10xf32>, %k : i64) {axis = 0 : i64, largest = true, sorted = true} -> tensor<10x10xf32>, tensor<1xi64>
    return
}

// -----

func.func @topk_negative_axis(%arg0: tensor<10x10xf32>) {
    %k = arith.constant 100 : i64
    // expected-error@+2 {{failed to infer returned types}}
    // expected-error@+1 {{expected axis to be within [-rank,rank) (where rank is the rank of the input)}}
    %a, %b = xten_nn.topk(%arg0 : tensor<10x10xf32>, %k : i64) {axis = -3 : i64, largest = true, sorted = true} -> tensor<10x10xf32>, tensor<1xi64>
    return
}

// -----

func.func @resize_5d_with_4_scales(%arg0: tensor<2x16x16x64x80xbf16>) -> tensor<2x16x32x128x160xbf16> {
    // expected-error@+1 {{'xten_nn.resize' op 'scales' size must match input rank (5)}}
    %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0>} : (tensor<2x16x16x64x80xbf16>) -> tensor<2x16x32x128x160xbf16>
    return %0 : tensor<2x16x32x128x160xbf16>
}

// -----

func.func @resize_4d_with_5_scales(%arg0: tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16> {
    // expected-error@+1 {{'xten_nn.resize' op 'scales' size must match input rank (4)}}
    %0 = xten_nn.resize %arg0 {coordinate_transformation_mode = 2 : i64, mode = 0 : i64, nearest_mode = 0 : i64, scales = array<f32: 1.0, 1.0, 2.0, 2.0, 2.0>} : (tensor<1x256x16x16xbf16>) -> tensor<1x256x32x32xbf16>
    return %0 : tensor<1x256x32x32xbf16>
}
