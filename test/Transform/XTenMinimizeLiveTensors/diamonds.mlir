// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -xten-minimize-live -split-input-file | FileCheck %s

// A diamond shaped dependency graph, where the order is expected to change.
// CHECK-LABEL:     one_diamond
// CHECK:     "ConvRelu_0"
// CHECK:     "ConvRelu_1"
// CHECK:     "Conv_0"
// CHECK:     "ConvAddRelu_0"

func.func @one_diamond(%arg0: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %4: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %2 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %5 : tensor<1x256x56x56xf32>
}

// -----

// Same as one_diamond, but the order is already as expected.
// CHECK-LABEL:     one_rev_diamond
// CHECK:     "ConvRelu_0"
// CHECK:     "ConvRelu_1"
// CHECK:     "Conv_0"
// CHECK:     "ConvAddRelu_0"

func.func @one_rev_diamond(%arg0: tensor<1x4x224x224xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %4 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %5 : tensor<1x256x56x56xf32>
}

// -----

// Check that the sort is stable for two identical diamonds
// CHECK-LABEL:     double_identical_diamond
// CHECK:     "ConvRelu_a0"
// CHECK:     "ConvRelu_a1"
// CHECK:     "Conv_a0"
// CHECK:     "ConvAddRelu_a0"
// CHECK:     "ConvRelu_b0"
// CHECK:     "ConvRelu_b1"
// CHECK:     "Conv_b0"
// CHECK:     "ConvAddRelu_b0"
// CHECK:     "Add_0"

func.func @double_identical_diamond(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_a0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_a1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_a0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %4 : tensor<1x64x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_a0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %6 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_b0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %7 = xten_nn.subgraph (%arg3 = %6: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_b1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %8 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_b0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %9 = xten_nn.subgraph (%arg3 = %7: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %8 : tensor<1x64x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_b0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %10 = xten_nn.subgraph (%arg3 = %5: tensor<1x256x56x56xf32>, %arg4 = %9: tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "Add_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %10 : tensor<1x256x56x56xf32>
}

// -----



// Check that the sort is stable for two identical diamonds
// CHECK-LABEL:     swap_double_identical_diamond
// CHECK:     "ConvRelu_b0"
// CHECK:     "ConvRelu_b1"
// CHECK:     "Conv_b0"
// CHECK:     "ConvAddRelu_b0"
// CHECK:     "ConvRelu_a0"
// CHECK:     "ConvRelu_a1"
// CHECK:     "Conv_a0"
// CHECK:     "ConvAddRelu_a0"
// CHECK:     "Add_0"

func.func @swap_double_identical_diamond(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_b0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_b1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_b0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %4 : tensor<1x64x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_b0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %6 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_a0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %7 = xten_nn.subgraph (%arg3 = %6: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_a1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %8 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_a0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %9 = xten_nn.subgraph (%arg3 = %7: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %8 : tensor<1x64x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_a0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %10 = xten_nn.subgraph (%arg3 = %5: tensor<1x256x56x56xf32>, %arg4 = %9: tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "Add_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %10 : tensor<1x256x56x56xf32>
}

// -----

// Two diamond shaped dependency graph, where the order is expected to change.
// CHECK-LABEL:     two_diamond
// CHECK:     "ConvRelu_0"
// CHECK:     "ConvRelu_1"
// CHECK:     "Conv_0"
// CHECK:     "ConvAddRelu_0"
// CHECK:     "ConvRelu_2"
// CHECK:     "ConvRelu_3"
// CHECK:     "Conv_1"
// CHECK:     "ConvAddRelu_1"

func.func @two_diamond(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %4: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %2 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %6 = xten_nn.subgraph (%arg3 = %5: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %7 = xten_nn.subgraph (%arg3 = %5: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_2", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %8 = xten_nn.subgraph (%arg3 = %7: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_3", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %9 = xten_nn.subgraph (%arg3 = %8: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %6 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_1", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %9 : tensor<1x256x56x56xf32>
}
