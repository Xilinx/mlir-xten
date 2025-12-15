// (c) Copyright 2025 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: aten-opt %s --split-input-file --mlir-print-debuginfo | FileCheck %s

func.func @subgraph_with_multi_attr_dict_loc(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %out = xten_nn.subgraph (%input = %arg0 : tensor<4x4xf32>) {
        xten_nn.output %input : tensor<4x4xf32>
    } -> tensor<4x4xf32> loc(#loc1)
    return %out : tensor<4x4xf32>
}
#loc1 = #xten_nn<dict_loc({OutputPath = "/someOutput", SubgraphName = "Conv2D"})>

// CHECK-LABEL: func.func @subgraph_with_multi_attr_dict_loc
// CHECK:       xten_nn.subgraph
// CHECK:       } -> tensor<4x4xf32> loc(#[[LOC1:loc.*]])
// CHECK-DAG:   #[[LOC1]] = #xten_nn<dict_loc({OutputPath = "/someOutput", SubgraphName = "Conv2D"})>
// -----

func.func @subgraph_with_fused_loc(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %result = xten_nn.subgraph (%x = %arg0 : tensor<8xf32>) {
        xten_nn.output %x : tensor<8xf32>
    } -> tensor<8xf32> loc(#loc2)
    return %result : tensor<8xf32>
}
#loc2 = loc(fused["some_loc", #xten_nn<dict_loc({SubgraphName = "something"})>])

// CHECK-LABEL: func.func @subgraph_with_fused_loc
// CHECK:       xten_nn.subgraph
// CHECK:       } -> tensor<8xf32> loc(#[[LOC2:loc.*]])
// CHECK-DAG:   #[[NAMELOC:loc.*]] = loc("some_loc")
// CHECK-DAG:   #[[DICTLOC:loc.*]] = #xten_nn<dict_loc({SubgraphName = "something"})>
// CHECK-DAG:   #[[LOC2]] = loc(fused[#[[NAMELOC]], #[[DICTLOC]]])
