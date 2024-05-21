// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -xten-minimize-live -split-input-file -o /dev/null -verify-diagnostics

// Illegal dead code where (dead) operation ConvRelu_1 cannot be scheduled.

// expected-error@+1 {{function cannot be rescheduled due to illegal dead code, aborting}}
func.func @illegal_dead_code(%arg0: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
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
  // expected-error@+1 {{Illegal operation}}
  %4 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %2 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %5 : tensor<1x256x56x56xf32>
}

// -----

// CHECK-LABEL: unknown_operation

func.func @unknown_operation(%arg0: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
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
  // expected-error@+1 {{Unknown operation}}
  %4 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "ConvRelu_1", Reason = "SomeReason"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %2: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %4 : tensor<1x64x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "ConvAddRelu_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %5 : tensor<1x256x56x56xf32>
}
