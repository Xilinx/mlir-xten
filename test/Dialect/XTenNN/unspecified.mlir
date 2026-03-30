// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights reserved.
//
// RUN: aten-opt %s | aten-opt | FileCheck %s
// RUN: aten-opt %s --mlir-print-op-generic | aten-opt | FileCheck %s

// -----

// CHECK-LABEL: func.func @unspecified_tensor
func.func @unspecified_tensor() {
  %0 = xten_nn.unspecified : tensor<2xf32>
  // CHECK: xten_nn.unspecified : tensor<2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @kernel_with_unspecified_attr
func.func @kernel_with_unspecified_attr() {
  %0 = xten_nn.kernel "myKernel" () {marker = #xten_nn.unspecified} -> tensor<1xf32>
  // CHECK: marker = #xten_nn.unspecified
  return
}

// -----

// CHECK-LABEL: func.func @tosa_pad_with_unspecified
func.func @tosa_pad_with_unspecified(%arg0: tensor<4xf32>) -> tensor<6xf32> {
  %pad = "tosa.const_shape"() <{value = dense<[1, 1]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %pad_value = xten_nn.unspecified : tensor<f32>
  %0 = tosa.pad %arg0, %pad, %pad_value : (tensor<4xf32>, !tosa.shape<2>, tensor<f32>) -> tensor<6xf32>
  // CHECK: xten_nn.unspecified : tensor<f32>
  // CHECK: tosa.pad %arg0, %{{.*}}, %{{.*}} : (tensor<4xf32>, !tosa.shape<2>, tensor<f32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
