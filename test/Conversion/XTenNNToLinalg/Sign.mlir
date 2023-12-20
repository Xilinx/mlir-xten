// RUN: aten-opt --convert-xtennn-to-linalg %s | FileCheck %s

func.func @sign(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.unified.sign %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK-LABEL:   func.func @sign(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xf32>) outs(%[[VAL_1]] : tensor<1x10xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_6:.*]] = arith.cmpf one, %[[VAL_3]], %[[VAL_5]] : f32
// CHECK:             %[[VAL_7:.*]] = arith.uitofp %[[VAL_6]] : i1 to f32
// CHECK:             %[[VAL_8:.*]] = math.copysign %[[VAL_7]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_9:.*]] = arith.cmpf uno, %[[VAL_3]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_3]], %[[VAL_8]] : f32
// CHECK:             linalg.yield %[[VAL_10]] : f32
// CHECK:           } -> tensor<1x10xf32>
// CHECK:           return %[[VAL_11:.*]] : tensor<1x10xf32>
}

// -----

func.func @sign_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    %0 = xten_nn.unified.sign %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
// CHECK-LABEL:   func.func @sign_int(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<1x10xi4>) -> tensor<1x10xi4> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xi4>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xi4>) outs(%[[VAL_1]] : tensor<1x10xi4>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i4, %[[VAL_4:.*]]: i4):
// CHECK:             %[[VAL_5:.*]] = arith.constant 0 : i4
// CHECK:             %[[VAL_6:.*]] = arith.constant 3 : i4
// CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i4
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_5]] : i4
// CHECK:             %[[VAL_9:.*]] = arith.shrsi %[[VAL_3]], %[[VAL_6]] : i4
// CHECK:             %[[VAL_10:.*]] = arith.ori %[[VAL_9]], %[[VAL_7]] : i4
// CHECK:             %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_5]], %[[VAL_10]] : i4
// CHECK:             linalg.yield %[[VAL_11]] : i4
// CHECK:           } -> tensor<1x10xi4>
// CHECK:           return %[[VAL_12:.*]] : tensor<1x10xi4>
}

// -----

func.func @sign_complex(%arg0: tensor<1x10xcomplex<f32>>) -> tensor<1x10xcomplex<f32>> {
    %0 = xten_nn.unified.sign %arg0 : (tensor<1x10xcomplex<f32>>) -> tensor<1x10xcomplex<f32>>
    return %0 : tensor<1x10xcomplex<f32>>
// CHECK-LABEL:   func.func @sign_complex(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x10xcomplex<f32>>) -> tensor<1x10xcomplex<f32>> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xcomplex<f32>>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xcomplex<f32>>) outs(%[[VAL_1]] : tensor<1x10xcomplex<f32>>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: complex<f32>, %[[VAL_4:.*]]: complex<f32>):
// CHECK:             %[[VAL_5:.*]] = complex.sign %[[VAL_3]] : complex<f32>
// CHECK:             linalg.yield %[[VAL_5]] : complex<f32>
// CHECK:           } -> tensor<1x10xcomplex<f32>>
// CHECK:           return %[[VAL_6:.*]] : tensor<1x10xcomplex<f32>>
}