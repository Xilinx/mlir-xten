// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -split-input-file | FileCheck %s

// Test backward compatibility (existing usage without method attribute)
// CHECK-LABEL: @test_load_external_const_original
func.func @test_load_external_const_original() -> tensor<4xf32> {
  // CHECK: xten_nn.load_external_const {file = "weights.h5", key = "layer1.weight"} -> tensor<4xf32>
  %0 = xten_nn.load_external_const {file = "weights.h5", key = "layer1.weight"} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Test new method attribute
// CHECK-LABEL: @test_load_external_const_with_method
func.func @test_load_external_const_with_method() -> tensor<4xf32> {
  // CHECK: xten_nn.load_external_const {file = "weights.h5", key = "layer1.weight", method = "hdf5"} -> tensor<4xf32>
  %0 = xten_nn.load_external_const {file = "weights.h5", key = "layer1.weight", method = "hdf5"} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Test with different method values
// CHECK-LABEL: @test_load_external_const_different_method
func.func @test_load_external_const_different_method() -> tensor<2x3xi32> {
  // CHECK: xten_nn.load_external_const {file = "data.bin", key = "conv.bias", method = "binary"} -> tensor<2x3xi32>
  %0 = xten_nn.load_external_const {file = "data.bin", key = "conv.bias", method = "binary"} -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// Test with complex tensor types
// CHECK-LABEL: @test_load_external_const_complex_type
func.func @test_load_external_const_complex_type() -> tensor<1x256x512xbf16> {
  // CHECK: xten_nn.load_external_const {file = "model.weights", key = "transformer.embeddings", method = "safetensors"} -> tensor<1x256x512xbf16>
  %0 = xten_nn.load_external_const {file = "model.weights", key = "transformer.embeddings", method = "safetensors"} -> tensor<1x256x512xbf16>
  return %0 : tensor<1x256x512xbf16>
}
