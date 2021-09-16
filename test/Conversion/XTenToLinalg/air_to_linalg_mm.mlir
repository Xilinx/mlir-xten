// RUN: aten-opt %s -xten-to-linalg | FileCheck %s
// CHECK: %2 = memref.alloc() : memref<32x32xf32>
// CHECK: linalg.matmul {__internal_linalg_transform__ = "ACDC_mmult"} ins(%0, %1 : memref<32x32xf32>, memref<32x32xf32>) outs(%2 : memref<32x32xf32>)
module  {
  func @myFunc(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "xten.mm"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
