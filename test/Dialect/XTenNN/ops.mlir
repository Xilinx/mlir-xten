// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s | aten-opt | FileCheck %s
// RUN: aten-opt %s --mlir-print-op-generic | aten-opt | FileCheck %s

// -----
// CHECK-LABEL: xten_nn.subgraph
func.func @subgraph(%arg0:  tensor<2xi64>) ->  tensor<2xi64> {
    %sum = xten_nn.subgraph (%c0 = %arg0 : tensor<2xi64>) {
        %sum = arith.addi %c0, %c0 :  tensor<2xi64>
        xten_nn.output %sum :  tensor<2xi64>
    } -> tensor<2xi64>
    return %sum :  tensor<2xi64>
}
// -----
// CHECK-LABEL: xten_nn.subgraph
func.func @subgraph_empty(%arg0:  tensor<2xi64>) ->  tensor<2xi64> {
    %sum = xten_nn.subgraph (%arg0 : tensor<2xi64>) -> tensor<2xi64>
    return %sum :  tensor<2xi64>
}


// -----

// CHECK-LABEL: kernel
func.func @kernel(%arg0: tensor<2xi64>, %arg1 : tensor<4xi64>) {
    xten_nn.kernel "myKernel" ()
    // CHECK: xten_nn.kernel "myKernel" ()
    %a = xten_nn.kernel "myKernel" () -> tensor<2xi64>
    // CHECK: xten_nn.kernel "myKernel" () -> tensor<2xi64>
    %b = xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>) -> tensor<2xi64>
    // CHECK: xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>) -> tensor<2xi64>
    %c = xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>) {attr = 4 : i32} -> tensor<2xi64>
    // CHECK: xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>) {attr = 4 : i32} -> tensor<2xi64>
    %d:2 = xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>, %arg1 : tensor<4xi64>) -> tensor<2xi64>, tensor<1xi64>
    // CHECK: xten_nn.kernel "myKernel" (%arg0 : tensor<2xi64>, %arg1 : tensor<4xi64>) -> tensor<2xi64>, tensor<1xi64>
    return
}

// -----

// CHECK-LABEL: topk
func.func @topk(%arg0: tensor<10x8xf32>) {
    %k = arith.constant 7 : i64
    // CHECK: %[[C7:.*]] = arith.constant 7 : i64
    xten_nn.topk(%arg0 : tensor<10x8xf32>, %k : i64) {axis = 0 : i64, largest = true, sorted = true} -> tensor<7x8xf32>, tensor<7x8xi64>
    // CHECK: xten_nn.topk(%arg0 : tensor<10x8xf32>, %[[C7]] : i64) {axis = 0 : i64, largest = true, sorted = true} -> tensor<7x8xf32>, tensor<7x8xi64>
    xten_nn.topk(%arg0 : tensor<10x8xf32>, %k : i64) {axis = 1 : i64, largest = true, sorted = true} -> tensor<10x7xf32>, tensor<10x7xi64>
    // CHECK: xten_nn.topk(%arg0 : tensor<10x8xf32>, %[[C7]] : i64) {axis = 1 : i64, largest = true, sorted = true} -> tensor<10x7xf32>, tensor<10x7xi64>
    return
}

// -----

// CHECK-LABEL: topk_arg
func.func @topk_arg(%arg0: tensor<10x8xf32>, %k: i64) {
    xten_nn.topk(%arg0 : tensor<10x8xf32>, %k : i64) {axis = 1 : i64, largest = true, sorted = true} -> tensor<10x?xf32>, tensor<10x?xi64>
    // CHECK: xten_nn.topk(%arg0 : tensor<10x8xf32>, %arg1 : i64) {axis = 1 : i64, largest = true, sorted = true} -> tensor<10x?xf32>, tensor<10x?xi64>
    return
}
