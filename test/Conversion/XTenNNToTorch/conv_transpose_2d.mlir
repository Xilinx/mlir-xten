// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @with_output_shape(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x72x44xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 1>, array<i64: 1, 1>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x72x44xf32>
  return %0 : tensor<1x360x72x44xf32>
}

// CHECK-LABEL:   func.func @with_output_shape(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                                 %[[VAL_2:.*]]: tensor<360xf32>) -> tensor<1x360x72x44xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<360xf32> -> !torch.vtensor<[360],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,36,22],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[360],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,360,72,44],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[1,360,72,44],f32> -> tensor<1x360x72x44xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<1x360x72x44xf32>

// -----

func.func @without_output_shape(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x74x46xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x74x46xf32>
  return %0 : tensor<1x360x74x46xf32>
}

// CHECK-LABEL:   func.func @without_output_shape(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: tensor<360xf32>) -> tensor<1x360x74x46xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<360xf32> -> !torch.vtensor<[360],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,36,22],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[360],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,360,74,46],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[1,360,74,46],f32> -> tensor<1x360x74x46xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<1x360x74x46xf32>

// -----

func.func @with_output_padding(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x75x47xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 1, 1>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x75x47xf32>
  return %0 : tensor<1x360x75x47xf32>
}

// CHECK-LABEL:   func.func @with_output_padding(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: tensor<360xf32>) -> tensor<1x360x75x47xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<360xf32> -> !torch.vtensor<[360],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,36,22],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[360],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,360,75,47],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[1,360,75,47],f32> -> tensor<1x360x75x47xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<1x360x75x47xf32>

// -----

func.func @with_pad_and_output_pad(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x73x45xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 1, 1>, pad = [array<i64: 1, 1>, array<i64: 1, 1>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x73x45xf32>
  return %0 : tensor<1x360x73x45xf32>
}

// CHECK-LABEL:   func.func @with_pad_and_output_pad(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: tensor<360xf32>) -> tensor<1x360x73x45xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<360xf32> -> !torch.vtensor<[360],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,36,22],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[360],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,360,73,45],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[1,360,73,45],f32> -> tensor<1x360x73x45xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<1x360x73x45xf32>

// -----

func.func @other_configs(%arg0: tensor<2x48x27x52xf32>, %arg1: tensor<48x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<2x36x84x161xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<2x48x27x52xf32>, tensor<48x36x6x8xf32>, tensor<36xf32>) -> tensor<2x36x84x161xf32>
  return %0 : tensor<2x36x84x161xf32>
}

// CHECK-LABEL:   func.func @other_configs(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<2x48x27x52xf32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<48x36x6x8xf32>,
// CHECK-SAME:                             %[[VAL_2:.*]]: tensor<36xf32>) -> tensor<2x36x84x161xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<2x48x27x52xf32> -> !torch.vtensor<[2,48,27,52],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<48x36x6x8xf32> -> !torch.vtensor<[48,36,6,8],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<36xf32> -> !torch.vtensor<[36],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[2,48,27,52],f32>, !torch.vtensor<[48,36,6,8],f32>, !torch.vtensor<[36],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[2,36,84,161],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[2,36,84,161],f32> -> tensor<2x36x84x161xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<2x36x84x161xf32>

// -----

func.func @other_configs_with_output_shape(%arg0: tensor<2x48x27x52xf32>, %arg1: tensor<48x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<2x36x82x162xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 1>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<2x48x27x52xf32>, tensor<48x36x6x8xf32>, tensor<36xf32>) -> tensor<2x36x82x162xf32>
  return %0 : tensor<2x36x82x162xf32>
}

// CHECK-LABEL:   func.func @other_configs_with_output_shape(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<2x48x27x52xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<48x36x6x8xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: tensor<36xf32>) -> tensor<2x36x82x162xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<2x48x27x52xf32> -> !torch.vtensor<[2,48,27,52],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<48x36x6x8xf32> -> !torch.vtensor<[48,36,6,8],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<36xf32> -> !torch.vtensor<[36],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_18:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[2,48,27,52],f32>, !torch.vtensor<[48,36,6,8],f32>, !torch.vtensor<[36],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[2,36,82,162],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[2,36,82,162],f32> -> tensor<2x36x82x162xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<2x36x82x162xf32>

// -----

func.func @dynamic_shape(%arg0: tensor<?x?x27x52xf32>, %arg1: tensor<?x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<?x?x?x?xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<?x?x27x52xf32>, tensor<?x36x6x8xf32>, tensor<36xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL:   func.func @dynamic_shape(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<?x?x27x52xf32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<?x36x6x8xf32>,
// CHECK-SAME:                             %[[VAL_2:.*]]: tensor<36xf32>) -> tensor<?x?x?x?xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<?x?x27x52xf32> -> !torch.vtensor<[?,?,27,52],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<?x36x6x8xf32> -> !torch.vtensor<[?,36,6,8],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<36xf32> -> !torch.vtensor<[36],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_21:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_20]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[?,?,27,52],f32>, !torch.vtensor<[?,36,6,8],f32>, !torch.vtensor<[36],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK-NEXT:          %[[VAL_22:.*]] = torch_c.to_builtin_tensor %[[VAL_21]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK-NEXT:          return %[[VAL_22]] : tensor<?x?x?x?xf32>

// -----

func.func @group_more_than_1(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<720xf32>) -> tensor<1x720x72x44xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 2 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 1>, array<i64: 1, 1>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<720xf32>) -> tensor<1x720x72x44xf32>
  return %0 : tensor<1x720x72x44xf32>
}

// CHECK-LABEL:   func.func @group_more_than_1(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                                 %[[VAL_2:.*]]: tensor<720xf32>) -> tensor<1x720x72x44xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<720xf32> -> !torch.vtensor<[720],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_18:.*]] = torch.prim.ListConstruct %[[VAL_16]], %[[VAL_17]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_19:.*]] = torch.aten.conv_transpose2d.input %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_18]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,36,22],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[720],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,720,72,44],f32>
// CHECK-NEXT:          %[[VAL_20:.*]] = torch_c.to_builtin_tensor %[[VAL_19]] : !torch.vtensor<[1,720,72,44],f32> -> tensor<1x720x72x44xf32>
// CHECK-NEXT:          return %[[VAL_20]] : tensor<1x720x72x44xf32>

// -----

func.func @asymmetric_pad(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<720xf32>) -> tensor<1x720x72x44xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 2 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 0>, array<i64: 1, 0>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<720xf32>) -> tensor<1x720x72x44xf32>
  return %0 : tensor<1x720x72x44xf32>
}

// CHECK-LABEL:   func.func @asymmetric_pad(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x480x36x22xf32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<480x360x4x4xf32>,
// CHECK-SAME:                              %[[VAL_2:.*]]: tensor<720xf32>) -> tensor<1x720x72x44xf32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
// CHECK-DAG:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x480x36x22xf32> -> !torch.vtensor<[1,480,36,22],f32>
// CHECK-DAG:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<480x360x4x4xf32> -> !torch.vtensor<[480,360,4,4],f32>
// CHECK-DAG:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<720xf32> -> !torch.vtensor<[720],f32>
// CHECK-DAG:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_10:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_13:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_15:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_14]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_18:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_19:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_20:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_22:.*]] = torch.aten.constant_pad_nd %[[VAL_3]], %[[VAL_21]], %[[VAL_16]] : !torch.vtensor<[1,480,36,22],f32>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,480,37,23],f32>
// CHECK-DAG:           %[[VAL_23:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_24:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_25:.*]] = torch.prim.ListConstruct %[[VAL_23]], %[[VAL_24]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:          %[[VAL_26:.*]] = torch.aten.conv_transpose2d.input %[[VAL_22]], %[[VAL_4]], %[[VAL_5]], %[[VAL_8]], %[[VAL_25]], %[[VAL_15]], %[[VAL_12]], %[[VAL_11]] : !torch.vtensor<[1,480,37,23],f32>, !torch.vtensor<[480,360,4,4],f32>, !torch.vtensor<[720],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,720,72,44],f32>
// CHECK-NEXT:          %[[VAL_27:.*]] = torch_c.to_builtin_tensor %[[VAL_26]] : !torch.vtensor<[1,720,72,44],f32> -> tensor<1x720x72x44xf32>
// CHECK-NEXT:          return %[[VAL_27]] : tensor<1x720x72x44xf32>
