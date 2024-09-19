// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s
// REQUIRES: torch

func.func @reduce_mean_one_axis_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 2>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32>
  return %0 : tensor<4x512x1x8xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_one_axis_keep_dims(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,1,8],f32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_6]] : !torch.vtensor<[4,512,1,8],f32> -> tensor<4x512x1x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<4x512x1x8xf32>
// CHECK:         }

// -----

func.func @reduce_mean_three_axes_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32>
  return %0 : tensor<4x1x1x1xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_three_axes_keep_dims(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_7]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1,1],f32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %[[VAL_8]] : !torch.vtensor<[4,1,1,1],f32> -> tensor<4x1x1x1xf32>
// CHECK:           return %[[VAL_9]] : tensor<4x1x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_mean_all_axes_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_all_axes_keep_dims(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1,1],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %[[VAL_9]] : !torch.vtensor<[1,1,1,1],f32> -> tensor<1x1x1x1xf32>
// CHECK:           return %[[VAL_10]] : tensor<1x1x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_mean_one_axis(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 2>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32>
  return %0 : tensor<4x512x8xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_one_axis(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,8],f32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_6]] : !torch.vtensor<[4,512,8],f32> -> tensor<4x512x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<4x512x8xf32>
// CHECK:         }

// -----

func.func @reduce_mean_three_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_three_axes(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_7]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4],f32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %[[VAL_8]] : !torch.vtensor<[4],f32> -> tensor<4xf32>
// CHECK:           return %[[VAL_9]] : tensor<4xf32>
// CHECK:         }

// -----

func.func @reduce_mean_all_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<f32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func.func @reduce_mean_all_axes(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %[[VAL_9]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[VAL_10]] : tensor<f32>
// CHECK:         }

// -----

func.func @reduce_mean_noop_with_empty_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32>
  return %0 : tensor<4x512x256x8xf32>
}

// CHECK-LABEL:   func.func @reduce_mean_noop_with_empty_axes(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_5]] : !torch.vtensor<[4,512,256,8],f32> -> tensor<4x512x256x8xf32>
// CHECK:           return %[[VAL_6]] : tensor<4x512x256x8xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_one_axis_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 2>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32>
  return %0 : tensor<4x512x1x8xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_one_axis_keep_dims(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x1x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,1,8],f32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_6]] : !torch.vtensor<[4,512,1,8],f32> -> tensor<4x512x1x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<4x512x1x8xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_three_axes_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32>
  return %0 : tensor<4x1x1x1xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_three_axes_keep_dims(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x1x1x1xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_7]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1,1],f32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %[[VAL_8]] : !torch.vtensor<[4,1,1,1],f32> -> tensor<4x1x1x1xf32>
// CHECK:           return %[[VAL_9]] : tensor<4x1x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_all_axes_keep_dims(%arg0: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_all_axes_keep_dims(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<1x1x1x1xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1,1],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %[[VAL_9]] : !torch.vtensor<[1,1,1,1],f32> -> tensor<1x1x1x1xf32>
// CHECK:           return %[[VAL_10]] : tensor<1x1x1x1xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_one_axis(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 2>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32>
  return %0 : tensor<4x512x8xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_one_axis(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,8],f32>
// CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_6]] : !torch.vtensor<[4,512,8],f32> -> tensor<4x512x8xf32>
// CHECK:           return %[[VAL_7]] : tensor<4x512x8xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_three_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_three_axes(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_7]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4],f32>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %[[VAL_8]] : !torch.vtensor<[4],f32> -> tensor<4xf32>
// CHECK:           return %[[VAL_9]] : tensor<4xf32>
// CHECK:         }

// -----

func.func @reduce_meanv13_all_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<f32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 0 : i64} : (tensor<4x512x256x8xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_all_axes(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %[[VAL_9]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[VAL_10]] : tensor<f32>
// CHECK:         }

// -----

func.func @reduce_meanv13_noop_with_empty_axes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
  %0 = xten_nn.reduce_mean %arg0 {axes = array<i64: 0, 1, 2, 3>, keepdims = 1 : i64} : (tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32>
  return %0 : tensor<4x512x256x8xf32>
}

// CHECK-LABEL:   func.func @reduce_meanv13_noop_with_empty_axes(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<4x512x256x8xf32> -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.aten.mean.dim %[[VAL_1]], %[[VAL_8]], %[[VAL_3]], %[[VAL_2]] : !torch.vtensor<[4,512,256,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,512,256,8],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.to_builtin_tensor %[[VAL_9]] : !torch.vtensor<[4,512,256,8],f32> -> tensor<4x512x256x8xf32>
// CHECK:           return %[[VAL_10]] : tensor<4x512x256x8xf32>
// CHECK:         }