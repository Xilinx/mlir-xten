// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s -xten-minimize-live="dot-before-sched" -split-input-file 2>&1 | FileCheck %s --check-prefix=DOT_BEFORE
// RUN: aten-opt %s -xten-minimize-live="dot-after-sched" -split-input-file 2>&1 | FileCheck %s --check-prefix=DOT_AFTER
// RUN: aten-opt %s -xten-minimize-live="dot-sched-cost" -split-input-file 2>&1 | FileCheck %s --check-prefix=DOT_SCHED_COST


// DOT_BEFORE: subgraph cost
// DOT_BEFORE: InCoreChain_0
// DOT_BEFORE-NEXT: 4014080
// DOT_BEFORE: InCoreChain_1
// DOT_BEFORE-NEXT: 6422528
// DOT_BEFORE: InCoreChain_2
// DOT_BEFORE-NEXT: 6422528
// DOT_BEFORE: InCoreChain_A
// DOT_BEFORE-NEXT: 5046272
// DOT_BEFORE: InCoreChain_B
// DOT_BEFORE-NEXT: 14450688
// DOT_BEFORE: InCoreChain_C
// DOT_BEFORE-NEXT: 4259840
// DOT_BEFORE: InCoreChain_3
// DOT_BEFORE-NEXT: 17334272

// DOT_BEFORE: Order
// DOT_BEFORE-NEXT: op0 -> op1
// DOT_BEFORE-NEXT: op1 -> op2
// DOT_BEFORE-NEXT: op2 -> op3
// DOT_BEFORE-NEXT: op3 -> op4
// DOT_BEFORE-NEXT: op4 -> op5
// DOT_BEFORE-NEXT: op5 -> op6
// DOT_BEFORE-NEXT: op6 -> op7
// DOT_BEFORE-NEXT: op7 -> op8


// DOT_AFTER: subgraph cost
// DOT_AFTER: InCoreChain_0
// DOT_AFTER-NEXT: 4014080
// DOT_AFTER: InCoreChain_1
// DOT_AFTER-NEXT: 6422528
// DOT_AFTER: InCoreChain_C
// DOT_AFTER-NEXT: 4259840
// DOT_AFTER: InCoreChain_2
// DOT_AFTER-NEXT: 6422528
// DOT_AFTER: InCoreChain_A
// DOT_AFTER-NEXT: 5046272
// DOT_AFTER: InCoreChain_B
// DOT_AFTER-NEXT: 14450688
// DOT_AFTER: InCoreChain_3
// DOT_AFTER-NEXT: 17334272

// DOT_AFTER: Order
// DOT_AFTER-NEXT: op0 -> op1
// DOT_AFTER-NEXT: op1 -> op2
// DOT_AFTER-NEXT: op2 -> op3
// DOT_AFTER-NEXT: op3 -> op4
// DOT_AFTER-NEXT: op4 -> op5
// DOT_AFTER-NEXT: op5 -> op6
// DOT_AFTER-NEXT: op6 -> op7
// DOT_AFTER-NEXT: op7 -> op8


// DOT_SCHED_COST: subgraph cost
// DOT_SCHED_COST: InCoreChain_0
// DOT_SCHED_COST-NEXT: 4014080
// DOT_SCHED_COST: InCoreChain_1
// DOT_SCHED_COST-NEXT: 6422528
// DOT_SCHED_COST: InCoreChain_2
// DOT_SCHED_COST-NEXT: 6422528
// DOT_SCHED_COST: InCoreChain_A
// DOT_SCHED_COST-NEXT: 5046272
// DOT_SCHED_COST: InCoreChain_B
// DOT_SCHED_COST-NEXT: 14450688
// DOT_SCHED_COST: InCoreChain_C
// DOT_SCHED_COST-NEXT: 4259840
// DOT_SCHED_COST: InCoreChain_3
// DOT_SCHED_COST-NEXT: 17334272

// DOT_SCHED_COST: Dependences
// DOT_SCHED_COST-NEXT:   op0 -> op2
// DOT_SCHED_COST-NEXT:   3211264
// DOT_SCHED_COST:   op0 -> op1
// DOT_SCHED_COST-NEXT:   3211264
// DOT_SCHED_COST:   op1 -> op5
// DOT_SCHED_COST-NEXT:   3211264
// DOT_SCHED_COST:   op1 -> op4
// DOT_SCHED_COST-NEXT:   3211264
// DOT_SCHED_COST:   op2 -> op3
// DOT_SCHED_COST-NEXT:   3211264
// DOT_SCHED_COST:   op3 -> op6
// DOT_SCHED_COST-NEXT:   1835008
// DOT_SCHED_COST:   op4 -> op6
// DOT_SCHED_COST-NEXT:   11239424
// DOT_SCHED_COST:   op5 -> op6
// DOT_SCHED_COST-NEXT:   1048576

func.func @multi_incore_chains(%arg0: tensor<1x4x224x224xf32>, %arg1: tensor<1x4x224x224xf32>, %arg2: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
  %0 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
  %1 = "tosa.const"() {value = dense<2.000000e-02> : tensor<64xf32>} : () -> tensor<64xf32>
  %2 = xten_nn.subgraph (%arg3 = %arg0: tensor<1x4x224x224xf32>, %arg4 = %0: tensor<64x4x7x7xf32>, %arg5 = %1: tensor<64xf32>, %arg6 = %arg2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index, 3 : index], LayerName = "InCoreChain_0", OfmShare = 3 : index, Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x4x224x224xf32>, %arg8 = %arg4: tensor<64x4x7x7xf32>, %arg9 = %arg5: tensor<64xf32>)  attributes {LayerName = "Conv_1_0", Reason = "MllibKernel", compile_time_configurations = "Conv2D_ReLU", config_attrs = {act = "RELU", batch_size = 1 : i64}, current_data_format = "NCHW", data_format = "HCWN", mllib_ops = "Conv2D", run_time_parameters = {act = 1 : i64, conv_type = [0, 3, 4], ksize = 7 : i64, stride_log2 = 1 : i64}, vectorization_granularity = "C8"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %3 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_1", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x64x112x112xf32>)  attributes {LayerName = "Add_2", Reason = "MllibKernel"} {
      %8 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %8 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %4 = xten_nn.subgraph (%arg3 = %2: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_2", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg7 = %arg3: tensor<1x64x112x112xf32>)  attributes {LayerName = "Conv_2_0", Reason = "MllibKernel"} {
      %10 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %10 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  %5 = xten_nn.subgraph (%arg3 = %4: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_A", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg5 = %arg3: tensor<1x64x112x112xf32>)  attributes {LayerName = "Add_4", Reason = "MllibKernel"} {
      %8 = tensor.empty() : tensor<1x64x64x112xf32>
      xten_nn.output %8 : tensor<1x64x64x112xf32>
    } -> tensor<1x64x64x112xf32>
    xten_nn.output %7 : tensor<1x64x64x112xf32>
  } -> tensor<1x64x64x112xf32>
  %6 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_B", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x64x112x112xf32>)  attributes {LayerName = "Add_5", Reason = "MllibKernel"} {
      %9 = tensor.empty() : tensor<1x112x112x224xf32>
      xten_nn.output %9 : tensor<1x112x112x224xf32>
    } -> tensor<1x112x112x224xf32>
    xten_nn.output %7 : tensor<1x112x112x224xf32>
  } -> tensor<1x112x112x224xf32>
  %7 = xten_nn.subgraph (%arg3 = %3: tensor<1x64x112x112xf32>)  attributes {IfmOperands = [0 : index], LayerName = "InCoreChain_C", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x64x112x112xf32>)  attributes {LayerName = "Add_7", Reason = "MllibKernel"} {
      %9 = tensor.empty() : tensor<1x64x64x64xf32>
      xten_nn.output %9 : tensor<1x64x64x64xf32>
    } -> tensor<1x64x64x64xf32>
    xten_nn.output %7 : tensor<1x64x64x64xf32>
  } -> tensor<1x64x64x64xf32>
  %8 = xten_nn.subgraph (%arg5 = %5: tensor<1x64x64x112xf32>, %arg4 = %7: tensor<1x64x64x64xf32>, %arg3 = %6: tensor<1x112x112x224xf32>)  attributes {IfmOperands = [0 : index, 1 : index, 2 : index], LayerName = "InCoreChain_3", Reason = "InCoreChain"} {
    %7 = xten_nn.subgraph (%arg8 = %arg3: tensor<1x112x112x224xf32>, %arg9 = %arg4: tensor<1x64x64x64xf32>)  attributes {ChainedWith = "Conv_7", LayerName = "Add_9", Reason = "MllibKernel"} {
      %9 = tensor.empty() : tensor<1x64x112x112xf32>
      xten_nn.output %9 : tensor<1x64x112x112xf32>
    } -> tensor<1x64x112x112xf32>
    xten_nn.output %7 : tensor<1x64x112x112xf32>
  } -> tensor<1x64x112x112xf32>
  return %8 : tensor<1x64x112x112xf32>
}