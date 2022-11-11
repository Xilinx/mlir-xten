// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @missing_terminator() -> f64 {
    // expected-error@+1 {{enclave is missing terminator}}
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
    // expected-error@+1 {{does not match number of enclave results}}
    output %1 : f64
    } -> f64, f64
    return %result1 : f64
}

// -----
func.func @enclave_result_type_missmatch() -> f32 {
    %result1 = xten_nn.subgraph () {
        %1 = arith.constant 1.0 : f64
    // expected-error@+1 {{does not match enclave result type}}
    output %1 : f64
    } -> f32
    return %result1 : f32
}


