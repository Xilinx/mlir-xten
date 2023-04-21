// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @valid_quantize_op_signed(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi8> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = 0.125: f32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}

// -----

func.func @valid_quantize_op_unsigned(%arg0: tensor<1x2xf32>) -> tensor<1x2xui8> {
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = 0.125: f32} -> tensor<1x2xui8>
    return %result : tensor<1x2xui8>
}

// -----

func.func @invalid_scale(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi8> {
    // expected-error@+1 {{failed to satisfy constraint: float32 is a Power-Of-Two value}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = 0.135: f32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}

// -----

func.func @invalid_negative_scale(%arg0: tensor<1x2xf32>) -> tensor<1x2xsi8> {
    // expected-error@+1 {{failed to satisfy constraint: float32 is a Power-Of-Two value}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = -0.25: f32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}

// -----

func.func @invalid_tensor_signless(%arg0: tensor<1x2xf32>) -> tensor<1x2xi8> {
    // expected-error@+1 {{op result #0 must be signed-or-unsigned-tensor of signed integer or unsigned integer values, but got 'tensor<1x2xi8>'}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = 0.5: f32} -> tensor<1x2xi8>
    return %result : tensor<1x2xi8>
}

// -----

func.func @invalid_io_shapes(%arg0: tensor<1x2xf32>) -> tensor<1x3xsi8> {
    // expected-error@+1 {{op requires the same shape for all operands and results}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xf32>) {scale = 0.5: f32} -> tensor<1x3xsi8>
    return %result : tensor<1x3xsi8>
}

// -----

func.func @invalid_input_type(%arg0: tensor<1x2xi32>) -> tensor<1x2xsi8> {
    // expected-error@+1 {{op operand #0 must be tensor of 32-bit float values, but got 'tensor<1x2xi32>'}}
    %result = xten_nn.quantize (%arg0: tensor<1x2xi32>) {scale = 0.5: f32} -> tensor<1x2xsi8>
    return %result : tensor<1x2xsi8>
}