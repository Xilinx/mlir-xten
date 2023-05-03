// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @valid_quantize_op_signed(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi8> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -3: si32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}

// -----

func.func @valid_quantize_op_unsigned(%arg0: tensor<1x2xf32>) -> tensor<1x2xui8> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -3: si32} -> tensor<1x2xui8>
    return %result : tensor<1x2xui8>
}

// -----

func.func @valid_quantize_op_large_scale(%arg0: tensor<1x2xf32>) -> tensor<1x2xui8> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = 5: si32} -> tensor<1x2xui8>
    return %result : tensor<1x2xui8>
}

// -----

func.func @invalid_shift(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi8> {
    // expected-error@+1 {{'xten_nn.quantize' op attribute 'shift' failed to satisfy constraint: 32-bit signed integer attribute}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = 0.135: f32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}

// -----

func.func @invalid_tensor_signless(%arg0: tensor<1x2xf32>) -> tensor<1x2xi8> {
    // expected-error@+1 {{op result #0 must be signed-or-unsigned-tensor of signed integer or unsigned integer values, but got 'tensor<1x2xi8>'}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -1: si32} -> tensor<1x2xi8>
    return %result : tensor<1x2xi8>
}

// -----

func.func @invalid_io_shapes(%arg0: tensor<1x2xf32>) -> tensor<1x3xsi8> {
    // expected-error@+1 {{op all non-scalar operands/results must have the same shape and base type}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -1: si32} -> tensor<1x3xsi8>
    return %result : tensor<1x3xsi8>
}

// -----

func.func @invalid_input_type(%arg0: tensor<1x2xi32>) -> tensor<1x2xsi8> {
    // expected-error@+1 {{op operand #0 must be tensor of 32-bit float values, but got 'tensor<1x2xi32>'}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xi32>) {shift = -1: si32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}
// -----

func.func @different_bitwidth(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi3> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -1: si32} -> tensor<1x2xsi3>
    return %result : tensor<1x2xsi3>
}

// -----

func.func @sixteen_bitwidth(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi16> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {shift = -1: si32} -> tensor<1x2xsi16>
    return %result : tensor<1x2xsi16>
}