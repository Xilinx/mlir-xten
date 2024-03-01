
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

func.func @sin_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.sin %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// -----

func.func @mish_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    // expected-error@+1 {{op operand #0 must be tensor of floating-point values, but got 'tensor<1x10xi4>'}}
    %0 = xten_nn.mish %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}
