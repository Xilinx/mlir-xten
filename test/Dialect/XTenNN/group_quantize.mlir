// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file -verify-diagnostics

func.func @valid_quantize_op(%arg0: tensor<1x2xf32>, %scales: tensor<1x1xf32>, %zeros: tensor<1x1xi4>) -> tensor<1x2xi4> {
    %result = xten_nn.group_quantize (%arg0: tensor<1x2xf32>, %scales: tensor<1x1xf32>, %zeros: tensor<1x1xi4>) {bits = 4 : si32, max = 7 : si32, min = -8 : si32} -> tensor<1x2xi4>
    return %result : tensor<1x2xi4>
}
