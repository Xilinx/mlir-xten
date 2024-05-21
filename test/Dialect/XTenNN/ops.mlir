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
