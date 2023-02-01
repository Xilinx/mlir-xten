// RUN: aten-opt %s -xten-minimize-live -split-input-file | FileCheck %s


// // -----

// CHECK-LABEL: func.func @multi_incore_chains_multiple_ifms_ofm
// CHECK:     "InCoreChain_0"
// CHECK:     "InCoreChain_1"
// CHECK:     "InCoreChain_4"
// CHECK:     "InCoreChain_3"
// CHECK:     "InCoreChain_2"
// CHECK:     "InCoreChain_5"
// CHECK:     "InCoreChain_6"
func.func @multi_incore_chains_multiple_ifms_ofm(%arg0: tensor<1x4x224x224xf32>, %arg1: tensor<1x4x224x224xf32>, %arg2: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
  %0 = xten_nn.subgraph ()  attributes {Reason = "SourceOp", SourceOp = "onnx.Constant", SourceOpAttrs = {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>}, argsConstMapping = {}, argsMapping = {}} {
    %7 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
    xten_nn.output %7 : tensor<64x4x7x7xf32>
  } -> tensor<64x4x7x7xf32>
  %1 = xten_nn.subgraph ()  attributes {Reason = "SourceOp", SourceOp = "onnx.Constant", SourceOpAttrs = {value = dense<2.000000e-02> : tensor<64xf32>}, argsConstMapping = {}, argsMapping = {}} {
    %7 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
    xten_nn.output %7 : tensor<64xf32>
  } -> tensor<64xf32>
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