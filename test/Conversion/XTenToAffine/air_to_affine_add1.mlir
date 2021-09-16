// RUN: aten-opt %s --xten-to-affine | FileCheck %s
// CHECK: affine.for %arg1 = 0 to 4 {
// CHECK:   affine.for %arg2 = 0 to 4 {
// CHECK:   %3 = affine.load %1[%arg1, %arg2] : memref<4x4xf32>
// CHECK:   %cst_0 = constant 1.000000e+00 : f32
// CHECK:   %4 = addf %3, %cst_0 : f32
// CHECK:   affine.store %4, %0[%arg1, %arg2] : memref<4x4xf32>
module {
  func @graph(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = constant dense<1.000000e+00> : tensor<f32>
    %1 = "xten.add_constant"(%arg0, %0) : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
