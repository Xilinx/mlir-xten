// RUN: aten-opt --convert-xtennn-to-linalg %s | FileCheck %s

func.func @mish(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.mish %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
}

// CHECK-LABEL:  func.func @mish
// CHECK:    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : tensor<1x10xf32>) outs(%{{.*}} : tensor<1x10xf32>) {
// CHECK:    ^bb0(%[[IN:.+]]: f32, %{{.*}}: f32):
// CHECK:      %[[EXP:.+]] = math.exp %[[IN]] : f32
// CHECK:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:      %[[ADD:.+]] = arith.addf %[[ONE]], %[[EXP]] : f32
// CHECK:      %[[LOG:.+]] = math.log %[[ADD]] : f32
// CHECK:      %[[TANH:.+]] = math.tanh %[[LOG]] : f32
// CHECK:      %[[MUL:.+]] = arith.mulf %[[IN]], %5 : f32
// CHECK:      linalg.yield %[[MUL]] : f32
// CHECK:    } -> tensor<1x10xf32>

// -----

func.func @mish_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    %0 = xten_nn.mish %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
}

// CHECK-LABEL:   func.func @mish_int(
//   CHECK-NOT: linalg.generic
