// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @valid_LEC_op() -> tensor<1x2xf32> {
    %result = xten_nn.load_external_const { key = "myLayerName", file = "weights.data"} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}

// -----

func.func @valid_LEC_op_bf16() -> tensor<1x2xbf16> {
    %result = xten_nn.load_external_const { key = "myLayerName", file = "weights.data"} -> tensor<1x2xbf16>
    return %result : tensor<1x2xbf16>
}

// -----

func.func @invalid_key(%arg0: tensor<1x2xsi8>) -> tensor<1x2xf32> {
    // expected-error@+1 {{'xten_nn.load_external_const' op attribute 'key' failed to satisfy constraint: string attribute}}
    %result = xten_nn.load_external_const { key = 12.0 : f32, file = "weights.data"} -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
}
