// RUN: aten-opt --convert-xtennn-to-linalg -split-input-file %s | FileCheck %s

func.func @round(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @round(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xf32>) outs(%[[VAL_1]] : tensor<1x10xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_4:.*]] = math.roundeven %[[VAL_3]] : f32
// CHECK:             linalg.yield %[[VAL_4]] : f32
// CHECK:           } -> tensor<1x10xf32>
// CHECK:           return %[[VAL_2:.*]] : tensor<1x10xf32>
}

// -----

func.func @round_bf16(%arg0: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @round_bf16(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xbf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xbf16>) outs(%[[VAL_1]] : tensor<1x10xbf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: bf16, %[[VAL_4:.*]]: bf16):
// CHECK:             %[[VAL_4:.*]] = math.roundeven %[[VAL_3]] : bf16
// CHECK:             linalg.yield %[[VAL_4]] : bf16
// CHECK:           } -> tensor<1x10xbf16>
// CHECK:           return %[[VAL_2:.*]] : tensor<1x10xbf16>
}

// -----

func.func @round_int(%arg0: tensor<1x10xi4>) -> tensor<1x10xi4> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xi4>) -> tensor<1x10xi4>
    return %0 : tensor<1x10xi4>
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @round_int(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<1x10xi4>) -> tensor<1x10xi4> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x10xi4>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x10xi4>) outs(%[[VAL_1]] : tensor<1x10xi4>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i4, %[[VAL_4:.*]]: i4):
// CHECK:             linalg.yield %[[VAL_3]] : i4
// CHECK:           } -> tensor<1x10xi4>
// CHECK:           return %[[VAL_2:.*]] : tensor<1x10xi4>
}
