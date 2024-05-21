// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @missing_terminator() -> f64 {
    // expected-error@+1 {{missing terminator}}
    %result = xten_nn.subgraph () {
        %1 = arith.constant 1.0 : f64
    } -> f64
    return %result : f64
}

// -----
func.func @enclave_result_number_missmatch() -> f64 {
    %result1, %result2 = xten_nn.subgraph () {
        %1 = arith.constant 1.0 : f64
        %2 = arith.constant 1.0 : f64
    // expected-error@+1 {{does not match number of results}}
    xten_nn.output %1 : f64
    } -> f64, f64
    return %result1 : f64
}

// -----
func.func @enclave_result_type_missmatch() -> f32 {
    %result1 = xten_nn.subgraph () {
        %1 = arith.constant 1.0 : f64
    // expected-error@+1 {{does not match result type}}
    xten_nn.output %1 : f64
    } -> f32
    return %result1 : f32
}

// -----
func.func @enclave_result_type_missmatch() -> f64 {
    %0 = arith.constant 1.0 : f64
    %1 = arith.constant 1.0 : f64
    // expected-error@+1 {{does not match argument type}}
    %result1 = "xten_nn.subgraph"(%0, %1) ({
    ^bb0(%arg1: f64, %arg2: f32):
        %2 = arith.constant 1.0 : f64
    "xten_nn.output"(%1) : (f64) -> ()
    }) : (f64, f64) -> (f64)
    return %result1 : f64
}

// -----
func.func @enclave_result_type_missmatch() -> f64 {
    %0 = arith.constant 1.0 : f64
    %1 = arith.constant 1.0 : f64
    // expected-error@+1 {{does not match number of arguments}}
    %result1 = "xten_nn.subgraph"(%0, %1) ({
    ^bb0(%arg1: f64):
        %2 = arith.constant 1.0 : f64
        "xten_nn.output"(%1) : (f64) -> ()
    }) : (f64, f64) -> (f64)
    return %result1 : f64
}


