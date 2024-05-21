// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @valid_dequantize_op_signed(%arg0: tensor<1x2xi8>) -> tensor<1x2xf32> {
    %result = xten_nn.dequantize (%arg0: tensor<1x2xi8>) {shift = -3: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @valid_dequantize_op_unsigned(%arg0: tensor<1x2xui8>) -> tensor<1x2xf32> {
    %result = xten_nn.dequantize (%arg0: tensor<1x2xui8>) {shift = -3: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @valid_dequantize_op_large_scale(%arg0: tensor<1x2xui8>) -> tensor<1x2xf32> {
    %result = xten_nn.dequantize (%arg0: tensor<1x2xui8>) {shift = 5: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}


// -----

func.func @invalid_shift(%arg0: tensor<1x2xi8>) -> tensor<1x2xf32> {
    // expected-error@+1 {{'xten_nn.dequantize' op attribute 'shift' failed to satisfy constraint: 32-bit signed integer attribute}}
    %result = xten_nn.dequantize (%arg0: tensor<1x2xi8>) {shift = 0.135: f32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @invalid_tensor_signed(%arg0: tensor<1x2xsi8>) -> tensor<1x2xf32> {
    // expected-error@+1 {{op operand #0 must be signless-or-unsigned-tensor of signless integer or unsigned integer values, but got 'tensor<1x2xsi8>}}
    %result = xten_nn.dequantize (%arg0: tensor<1x2xsi8>) {shift = -1: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @invalid_io_shapes(%arg0: tensor<1x3xi8>) -> tensor<1x2xf32> {
    // expected-error@+1 {{op all non-scalar operands/results must have the same shape and base type}}
    %result = xten_nn.dequantize (%arg0: tensor<1x3xi8>) {shift = -1: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @invalid_output_type(%arg0: tensor<1x2xi8>) -> tensor<1x2xi32> {
    // expected-error@+1 {{op result #0 must be tensor of 32-bit float values, but got 'tensor<1x2xi32>'}}
    %result = xten_nn.dequantize (%arg0: tensor<1x2xi8>) {shift = -1: si32} -> tensor<1x2xi32>
    return %result : tensor<1x2xi32>
}

// -----

func.func @different_bitwidth(%arg0: tensor<1x2xi3>) -> tensor<1x2xf32> {
    %result = xten_nn.dequantize (%arg0: tensor<1x2xi3>) {shift = -1: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @sixteen_bitwidth(%arg0: tensor<1x2xi16>) -> tensor<1x2xf32> {
    %result = xten_nn.dequantize (%arg0: tensor<1x2xi16>) {shift = -1: si32} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}