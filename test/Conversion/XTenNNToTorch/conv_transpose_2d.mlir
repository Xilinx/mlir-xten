// RUN: aten-opt --convert-xtennn-to-torch  -split-input-file %s | FileCheck %s

func.func @with_output_shape(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x72x44xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 1>, array<i64: 1, 1>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x72x44xf32>
  return %0 : tensor<1x360x72x44xf32>
}

// -----

func.func @without_output_shape(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x74x46xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x74x46xf32>
  return %0 : tensor<1x360x74x46xf32>
}

// -----

func.func @with_output_padding(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x75x47xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 1, 1>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x75x47xf32>
  return %0 : tensor<1x360x75x47xf32>
}

// -----

func.func @with_pad_and_output_pad(%arg0: tensor<1x480x36x22xf32>, %arg1: tensor<480x360x4x4xf32>, %arg2: tensor<360xf32>) -> tensor<1x360x73x45xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 1, 1>, pad = [array<i64: 1, 1>, array<i64: 1, 1>], stride = array<i64: 2, 2>} : (tensor<1x480x36x22xf32>, tensor<480x360x4x4xf32>, tensor<360xf32>) -> tensor<1x360x73x45xf32>
  return %0 : tensor<1x360x73x45xf32>
}

// -----

func.func @other_configs(%arg0: tensor<2x48x27x52xf32>, %arg1: tensor<48x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<2x36x84x161xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<2x48x27x52xf32>, tensor<48x36x6x8xf32>, tensor<36xf32>) -> tensor<2x36x84x161xf32>
  return %0 : tensor<2x36x84x161xf32>
}

// -----

func.func @other_configs_with_output_shape(%arg0: tensor<2x48x27x52xf32>, %arg1: tensor<48x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<2x36x82x162xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 1, 1>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<2x48x27x52xf32>, tensor<48x36x6x8xf32>, tensor<36xf32>) -> tensor<2x36x82x162xf32>
  return %0 : tensor<2x36x82x162xf32>
}

// -----

func.func @dynamic_shape_results(%arg0: tensor<2x48x27x52xf32>, %arg1: tensor<48x36x6x8xf32>, %arg2: tensor<36xf32>) -> tensor<?x?x?x?xf32> {
  %0 = xten_nn.ConvTranspose %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, group = 1 : i64, output_padding = array<i64: 0, 0>, pad = [array<i64: 0, 0>, array<i64: 0, 0>], stride = array<i64: 3, 3>} : (tensor<2x48x27x52xf32>, tensor<48x36x6x8xf32>, tensor<36xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
