// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -xten-minimize-live -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @multi_incore_chains_multiple_ifms_ofm
// CHECK:     "InCoreChain_0"
// CHECK:     "InCoreChain_1"
// CHECK:     "InCoreChain_4"
// CHECK:     "InCoreChain_3"
// CHECK:     "InCoreChain_2"
// CHECK:     "InCoreChain_5"
// CHECK:     "InCoreChain_6"
func.func @multi_incore_chains_multiple_ifms_ofm(%arg0: tensor<1x4x224x224xf32>, %arg1: tensor<1x4x224x224xf32>, %arg2: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_1_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_1_1", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %9 = xten_nn.subgraph (%arg7 = %7: tensor<1x64x112x112xf32>, %arg8 = %arg6: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_1", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %9 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %3 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x112x112xf32>, %arg4 = %2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x64x112x112xf32>, %arg6 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_2", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %8 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %8 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_2", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_1", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %9 = xten_nn.subgraph (%arg7 = %7: tensor<1x64x112x112xf32>, %arg8 = %arg6: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_3", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %9 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %5 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x112x112xf32>, %arg4 = %2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "InCoreChain_3", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x64x112x112xf32>, %arg6 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_4", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %8 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %8 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %6 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x112x112xf32>, %arg4 = %3: tensor<1x64x112x112xf32>, %arg5 = %0: tensor<64x4x7x7xf32>, %arg6 = %1: tensor<64xf32>, %arg7 = %arg0: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index, 4 : index], LayerName = "InCoreChain_4", OfmShare = 0 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_5", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg8 = %7: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_6", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %8 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %7 = xten_nn.subgraph (%arg3 = %4: tensor<1x64x112x112xf32>, %arg4 = %5: tensor<1x64x112x112xf32>, %arg5 = %0: tensor<64x4x7x7xf32>, %arg6 = %1: tensor<64xf32>, %arg7 = %arg0: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index, 4 : index], LayerName = "InCoreChain_5", OfmShare = 0 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_7", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg8 = %7: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_8", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %8 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %8 = xten_nn.subgraph (%arg3 = %6: tensor<1x64x112x112xf32>, %arg4 = %7: tensor<1x64x112x112xf32>, %arg5 = %0: tensor<64x4x7x7xf32>, %arg6 = %1: tensor<64xf32>, %arg7 = %arg0: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index, 4 : index], LayerName = "InCoreChain_6", OfmShare = 0 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_9", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg8 = %7: tensor<1x64x112x112xf32>, %arg9 = %arg4: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_10", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %8 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  return %8 : tensor<1x64x112x112xf32>
}

// -----

// CHECK-LABEL: func.func @incorechain_ops_with_concat
// CHECK:     "InCoreChain_0"
// CHECK:     "InCoreChain_1"
// CHECK:     "InCoreChain_2"
// CHECK:     "Concat0"
func.func @incorechain_ops_with_concat(%arg0: tensor<1x4x224x224xf32>, %arg1: tensor<1x4x224x224xf32>, %arg2: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_1_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x4x224x224xf32>
      xten_nn.output %10 : tensor<1x4x224x224xf32>
    } -> tensor<1x4x224x224xf32>
    xten_nn.output %7 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %3 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x4x224x224xf32>, %arg4 = %arg1: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x4x224x224xf32>, %arg6 = %arg4: tensor<1x4x224x224xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_2", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %8 = tensor.empty() : tensor<1x4x64x64xf32>
      xten_nn.output %8 : tensor<1x4x64x64xf32>
    } -> tensor<1x4x64x64xf32>
    xten_nn.output %7 : tensor<1x4x64x64xf32>
  } -> tensor<1x4x64x64xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_2", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_1", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %9 = xten_nn.subgraph (%arg7 = %7: tensor<1x64x112x112xf32>, %arg8 = %arg6: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_3", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %9 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %5 = xten_nn.subgraph (%arg1 = %2: tensor<1x4x224x224xf32>, %arg2 = %3: tensor<1x4x64x64xf32>, %arg3 = %4: tensor<1x64x112x112xf32>)  attributes {LayerName = "Concat0", Reason = "PseudoOp", Op = "Concat", SourceOpAttrs = {axis = 1 : si64, onnx_node_name = "Concat_0", output_bitwidth = 8.000000e+00 : f32, output_narrow = 0 : si64, output_rounding_mode = "ROUND", output_scale_factor = 2.500000e-01 : f32, output_signed = 1 : si64}, argsConstMapping = {}, argsMapping = {"0" = 0 : index, "1" = 1 : index}} {
      %89 = tensor.empty() : tensor<1x64x112x112xf32>
    xten_nn.output %89 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  return %5 : tensor<1x64x112x112xf32>
}

// -----

// Legal dead code where (dead) operation Conv_0 is not attached to any operation that influence output.
// The order is expected to change.

// CHECK-LABEL:     legal_dead_code
// CHECK:     "Conv_2"
// CHECK:     "Conv_1"
// CHECK:     "Conv_0"
// CHECK:     "Add_0"

func.func @legal_dead_code(%arg0: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_0", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  %3 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_1", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x256x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Conv_2", Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %8 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  %5 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x56x56xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %2 : tensor<1x256x56x56xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "Add_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %8 = tensor.empty() : tensor<1x256x56x56xf32>
    xten_nn.output %8 : tensor<1x256x56x56xf32>
  } -> tensor<1x256x56x56xf32>
  return %5 : tensor<1x256x56x56xf32>
}

// -----

// CHECK-LABEL: func.func @incorechain_ops_with_concat2
// CHECK:     "InCoreChain_0"
// CHECK:     "InCoreChain_1"
// CHECK:     "InCoreChain_2"
// CHECK:     "Concat0"
func.func @incorechain_ops_with_concat2(%arg0: tensor<1x4x224x224xf32>, %arg1: tensor<1x4x224x224xf32>, %arg2: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %4 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_2", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %8 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_2_1", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    %9 = xten_nn.subgraph (%arg7 = %7: tensor<1x64x112x112xf32>, %arg8 = %arg6: tensor<1x64x112x112xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_3", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %9 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_1_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x4x224x224xf32>
      xten_nn.output %10 : tensor<1x4x224x224xf32>
    } -> tensor<1x4x224x224xf32>
    xten_nn.output %7 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %3 = xten_nn.subgraph (%arg3 = %arg1: tensor<1x4x224x224xf32>, %arg4 = %arg1: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x4x224x224xf32>, %arg6 = %arg4: tensor<1x4x224x224xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_2", Reason = "MllibKernel", compile_time_configurations = "Add2d", config_attrs = {batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Add2d", run_time_parameters = {act = 1 : i64, batch_log2 = 0 : i64}, vectorization_granularity = "C8"} {
      %8 = tensor.empty() : tensor<1x4x64x64xf32>
      xten_nn.output %8 : tensor<1x4x64x64xf32>
    } -> tensor<1x4x64x64xf32>
    xten_nn.output %7 : tensor<1x4x64x64xf32>
  } -> tensor<1x4x64x64xf32>
  %5 = xten_nn.subgraph (%arg1 = %2: tensor<1x4x224x224xf32>, %arg2 = %3: tensor<1x4x64x64xf32>, %arg3 = %4: tensor<1x64x112x112xf32>)  attributes {LayerName = "Concat0", Reason = "PseudoOp", Op = "Concat", SourceOpAttrs = {axis = 1 : si64, onnx_node_name = "Concat_0", output_bitwidth = 8.000000e+00 : f32, output_narrow = 0 : si64, output_rounding_mode = "ROUND", output_scale_factor = 2.500000e-01 : f32, output_signed = 1 : si64}, argsConstMapping = {}, argsMapping = {"0" = 0 : index, "1" = 1 : index}} {
      %89 = tensor.empty() : tensor<1x64x112x112xf32>
    xten_nn.output %89 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  return %5 : tensor<1x64x112x112xf32>
}

// -----

// CHECK-LABEL: func.func @gap_concat
// CHECK: %[[GAP1:.*]] = xten_nn.subgraph {{.*}} LayerName = "GAP1"{{.*}} Reason = "InCoreChain"
// CHECK: %[[GAP0:.*]] = xten_nn.subgraph {{.*}} LayerName = "GAP0"{{.*}} Reason = "InCoreChain"
// CHECK: xten_nn.subgraph ({{.*}} = %[[GAP0]]{{.*}}, {{.*}} = %[[GAP1]]{{.*}}LayerName = "Concat0", Op = "Concat", Reason = "PseudoOp"

func.func @gap_concat(%arg0 : tensor<1x2x255x255xf32>, %arg1 : tensor<1x2x255x255xf32>) -> tensor<1x4x1x1xf32> {
  %0 = xten_nn.subgraph (%arg0 = %arg0 : tensor<1x2x255x255xf32>) attributes { IfmOperands = [0: index], LayerName = "GAP0", OutputName = "GAP0", Reason = "InCoreChain" } {
    %0 = xten_nn.subgraph () attributes { LayerName = "GAP0", OutputName = "GAP0", Reason = "MllibKernel", mllib_ops = "GlobalAvgPool2D" } {
      %0 = tensor.empty() : tensor<1x2x1x1xf32>
      xten_nn.output %0 : tensor<1x2x1x1xf32>
    } -> tensor<1x2x1x1xf32>
    xten_nn.output %0 : tensor<1x2x1x1xf32>
  } -> tensor<1x2x1x1xf32>
  %1 = xten_nn.subgraph (%arg0 = %arg1 : tensor<1x2x255x255xf32>) attributes { IfmOperands = [0: index], LayerName = "GAP1", OutputName = "GAP1", Reason = "InCoreChain" } {
    %0 = xten_nn.subgraph () attributes { LayerName = "GAP1", OutputName = "GAP1", Reason = "MllibKernel", mllib_ops = "GlobalAvgPool2D" } {
      %0 = tensor.empty() : tensor<1x2x1x1xf32>
      xten_nn.output %0 : tensor<1x2x1x1xf32>
    } -> tensor<1x2x1x1xf32>
    xten_nn.output %0 : tensor<1x2x1x1xf32>
  } -> tensor<1x2x1x1xf32>
  %2 = xten_nn.subgraph (%arg1 = %0: tensor<1x2x1x1xf32>, %arg2 = %1: tensor<1x2x1x1xf32>) attributes { LayerName = "Concat0", Reason = "PseudoOp", Op = "Concat" } {
      %0 = tensor.empty() : tensor<1x4x1x1xf32>
      xten_nn.output %0 : tensor<1x4x1x1xf32>
  } -> tensor<1x4x1x1xf32>
  return %2 : tensor<1x4x1x1xf32>
}

// -----

// CHECK-LABEL: func.func @gap_concat_reversed
// CHECK: %[[GAP0:.*]] = xten_nn.subgraph {{.*}} LayerName = "GAP0"{{.*}} Reason = "InCoreChain"
// CHECK: %[[GAP1:.*]] = xten_nn.subgraph {{.*}} LayerName = "GAP1"{{.*}} Reason = "InCoreChain"
// CHECK: xten_nn.subgraph ({{.*}} = %[[GAP1]]{{.*}}, {{.*}} = %[[GAP0]]{{.*}}LayerName = "Concat0", Op = "Concat", Reason = "PseudoOp"

func.func @gap_concat_reversed(%arg0 : tensor<1x2x255x255xf32>, %arg1 : tensor<1x2x255x255xf32>) -> tensor<1x4x1x1xf32> {
  %0 = xten_nn.subgraph (%arg0 = %arg0 : tensor<1x2x255x255xf32>) attributes { IfmOperands = [0: index], LayerName = "GAP0", OutputName = "GAP0", Reason = "InCoreChain" } {
    %0 = xten_nn.subgraph () attributes { LayerName = "GAP0", OutputName = "GAP0", Reason = "MllibKernel", mllib_ops = "GlobalAvgPool2D" } {
      %0 = tensor.empty() : tensor<1x2x1x1xf32>
      xten_nn.output %0 : tensor<1x2x1x1xf32>
    } -> tensor<1x2x1x1xf32>
    xten_nn.output %0 : tensor<1x2x1x1xf32>
  } -> tensor<1x2x1x1xf32>
  %1 = xten_nn.subgraph (%arg0 = %arg1 : tensor<1x2x255x255xf32>) attributes { IfmOperands = [0: index], LayerName = "GAP1", OutputName = "GAP1", Reason = "InCoreChain" } {
    %0 = xten_nn.subgraph () attributes { LayerName = "GAP1", OutputName = "GAP1", Reason = "MllibKernel", mllib_ops = "GlobalAvgPool2D" } {
      %0 = tensor.empty() : tensor<1x2x1x1xf32>
      xten_nn.output %0 : tensor<1x2x1x1xf32>
    } -> tensor<1x2x1x1xf32>
    xten_nn.output %0 : tensor<1x2x1x1xf32>
  } -> tensor<1x2x1x1xf32>
  %2 = xten_nn.subgraph (%arg1 = %1: tensor<1x2x1x1xf32>, %arg2 = %0: tensor<1x2x1x1xf32>) attributes { LayerName = "Concat0", Reason = "PseudoOp", Op = "Concat" } {
      %0 = tensor.empty() : tensor<1x4x1x1xf32>
      xten_nn.output %0 : tensor<1x4x1x1xf32>
  } -> tensor<1x4x1x1xf32>
  return %2 : tensor<1x4x1x1xf32>
}


// -----

// The order is already as expected.

// CHECK-LABEL: func.func @support_for_inteface_op
// CHECK: LayerName = "InCoreChain_0"{{.*}} Reason = "InCoreChain"
// CHECK: LayerName = "Interface_0"{{.*}} Reason = "Interface"
// CHECK: LayerName = "Interface_1"{{.*}} Reason = "Interface"
// CHECK: LayerName = "InCoreChain_1"{{.*}} Reason = "InCoreChain"
func.func @support_for_inteface_op(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
  %1 = xten_nn.subgraph (%arg1 = %arg0: tensor<1x3x224x224xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_0", OutputName = "MaxPool_147", Reason = "InCoreChain"} {
    %0 = tensor.empty() : tensor<1x4x224x224xf32>
    xten_nn.output %0 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %2 = xten_nn.subgraph (%arg1 = %1: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index], Reason = "Interface", LayerName = "Interface_0"} {
    %0 = tensor.empty() : tensor<1x5x224x224xf32>
    xten_nn.output %0 : tensor<1x5x224x224xf32>
  } -> tensor<1x5x224x224xf32>
  %3 = xten_nn.subgraph (%arg1 = %arg0: tensor<1x3x224x224xf32>)  attributes {IfmOperands = [0 : index], Reason = "Interface", LayerName = "Interface_1"} {
    %0 = tensor.empty() : tensor<1x4x224x224xf32>
    xten_nn.output %0 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %4 = xten_nn.subgraph (%arg1 = %0: tensor<64x3x7x7xf32>)  attributes {Reason = "Interface", LayerName = "Interface_2"} {
    %0 = tensor.empty() : tensor<64x4x7x7xf32>
    xten_nn.output %0 : tensor<64x4x7x7xf32>
  } -> tensor<64x4x7x7xf32>
  %5 = xten_nn.subgraph (%arg1 = %2: tensor<1x5x224x224xf32>, %arg2 = %4: tensor<64x4x7x7xf32>, %arg3 = %3: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index, 2 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %6 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %6 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  return %5 : tensor<1x64x56x56xf32>
}

// -----

// CHECK-LABEL: func.func @support_for_inteface_op
// CHECK: LayerName = "InCoreChain_0"{{.*}} Reason = "InCoreChain"
// CHECK: LayerName = "Transpose"{{.*}} Reason = "TemplatedGraph"
// CHECK: LayerName = "InCoreChain_1"{{.*}} Reason = "InCoreChain"
func.func @support_for_inteface_op(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x56x56xf32> {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
  %1 = xten_nn.subgraph (%arg1 = %arg0: tensor<1x3x224x224xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_0", OutputName = "Conv", Reason = "InCoreChain"} {
    %0 = tensor.empty() : tensor<1x4x224x224xf32>
    xten_nn.output %0 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %2 = xten_nn.subgraph (%arg1 = %1: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index], LayerName = "Transpose", OutputName = "TransposeAdf", Reason = "TemplatedGraph"} {
    %0 = tensor.empty() : tensor<1x4x224x224xf32>
    xten_nn.output %0 : tensor<1x4x224x224xf32>
  } -> tensor<1x4x224x224xf32>
  %3 = xten_nn.subgraph (%arg1 = %2: tensor<1x4x224x224xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %6 = tensor.empty() : tensor<1x64x56x56xf32>
    xten_nn.output %6 : tensor<1x64x56x56xf32>
  } -> tensor<1x64x56x56xf32>
  return %3 : tensor<1x64x56x56xf32>
}