// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file --xtennn-simplify | FileCheck %s

func.func @net0() -> f64 {
    %unused = arith.constant 0.0 : f64
    %result = xten_nn.subgraph (%0 = %unused : f64) {
        %1 = arith.constant 1.0 : f64
        xten_nn.output %1 : f64
    } -> f64
    return %result : f64
// CHECK-LABEL: @net0(
// CHECK: %[[result:.+]] = xten_nn.subgraph () {
// CHECK: } -> f64
}

// -----

func.func @net1() -> f64 {
    %r0, %r1 = xten_nn.subgraph () {
        %cst0 = arith.constant 0.0 : f64
        %cst1 = arith.constant 1.0 : f64
        xten_nn.output %cst0, %cst1 : f64, f64
    } -> f64, f64
    return %r0 : f64
// CHECK-LABEL: @net1(
// CHECK: %[[r0:.+]] = xten_nn.subgraph () {
// CHECK: %[[cst0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK: xten_nn.output %[[cst0]] : f64
// CHECK: } -> f64
// CHECK: return %[[r0]] : f64
}

