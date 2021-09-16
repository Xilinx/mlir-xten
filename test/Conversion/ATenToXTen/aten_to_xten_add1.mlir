// RUN: aten-opt %s --aten-to-xten | FileCheck %s

// CHECK: %[[VAL_1:.*]] = constant dense<1.111110e+00> : tensor<f32>
// CHECK: %[[VAL_2:.*]] = "xten.add_constant"(%{{.*}}, %[[VAL_1]]) : (tensor<256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
module {
  func @graph(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = constant dense<1.111110e+00> : tensor<f32>
    %c1_i64 = constant 1 : i64
    %0 = "aten.add"(%arg0, %cst, %c1_i64) : (tensor<256x256xf32>, tensor<f32>, i64) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
