
// RUN: aten-opt %s -xten-to-linalg='linalg-on-tensors=true' | FileCheck %s
// CHECK: %0 = linalg.init_tensor [128, 128] : tensor<128x128xi32>
// CHECK: %1 = linalg.matmul {{.*}} ins(%arg0, %arg1 : tensor<128x128xi32>, tensor<128x128xi32>) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
module  {
  func @task(%arg0: tensor<128x128xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = "xten.mm"(%arg0, %arg1) : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    return %0 : tensor<128x128xi32>
  }
}