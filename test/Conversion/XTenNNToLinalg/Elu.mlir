// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt --convert-xtennn-to-linalg %s | FileCheck %s

func.func @elu(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.elu %arg0 { alpha = 1.000000e-00 : f32} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
}

// CHECK-LABEL:  func.func @elu
// CHECK:    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : tensor<1x10xf32>) outs(%{{.*}} : tensor<1x10xf32>) {
// CHECK:    ^bb0(%[[IN:.+]]: f32, %{{.*}}: f32):
// CHECK:      %[[EXP:.+]] = math.exp %[[IN]] : f32
// CHECK:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:      %[[SUB:.+]] = arith.subf %[[EXP]], %[[ONE]] : f32
// CHECK:      %[[ALPHA:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:      %[[MUL:.+]] = arith.mulf %[[ALPHA]], %[[SUB]] : f32
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[CMP:.+]] = arith.cmpf ugt, %[[IN]], %[[ZERO]] : f32
// CHECK:      %[[ELU:.+]] = arith.select %[[CMP]], %[[IN]], %[[MUL]] : f32
// CHECK:      linalg.yield %[[ELU]] : f32
