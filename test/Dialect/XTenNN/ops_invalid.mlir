// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics


func.func @atan2_int(%arg0: tensor<1x10xi4>, %arg1: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.atan2 %arg0, %arg1 : (tensor<1x10xi4>, tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @cos_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.cos %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
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

func.func @round_int(%arg0: tensor<1x10xsi32>) -> tensor<1x10xsi32> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xsi32>'}}
    %0 = xten_nn.round %arg0 : (tensor<1x10xsi32>) -> tensor<1x10xsi32>
    return %0 : tensor<1x10xsi32>
}

// -----

func.func @sign_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.sign %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @sin_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.sin %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}