//===-- XTenNNPasses.td - XtenNN passes definitions *----- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the XTenNN dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_PASSES
#define XTENNN_PASSES

include "mlir/Pass/PassBase.td"


def XTenNNSimplify : Pass<"xtennn-simplify", "mlir::ModuleOp"> {
    let summary = "Simplify XTenNN subgraphs";
    let description = [{
        Simplifies `xten_nn.subgraph` contents using greedy rewrite patterns, which
        are semantically preserving but metadata destroying.

        Patterns:
            - RemoveUnusedCaptures

              Removes capture arguments from EnclaveOps that are unused.

              Example:
              ```mlir
              %2 = xten_nn.subgraph (%arg0 = %0 : f64, %arg1 = %1 : f64) {
                  output %arg1 : f64
              } -> f64
              ```
              is simplified to:
              ```mlir
              %2 = xten_nn.subgraph (%arg0 = %1 : f64) {
                  output %arg0 : f64
              } -> f64
              ```

            - RemoveUnusedReturns

              Removes EnclaveOp returns that are unused.

              Example:
              ```mlir
              func.func @my_func(%0: f64) -> f64 {
                  %1, %2 = xten_nn.subgraph (%arg0 = %0 : f64) {
                      %0 = arith.addf %arg0, %arg0 : f64
                      %1 = arith.mulf %arg0, %arg0 : f64
                      output %0, %1 : f64, f64
                  } -> f64, f64
                  return %1 : f64
              }
              ```
              is simplified to:
              ```mlir
              func.func @my_func(%0: f64) -> f64 {
                  %1 = xten_nn.subgraph (%arg0 = %0 : f64) {
                      %0 = arith.addf %arg0, %arg0 : f64
                      output %0 : f64
                  } -> f64
                  return %1 : f64
              }
              ```
    }];

    let constructor = "amd::xten_nn::createSimplifyPass()";
}

def XTenNNCanonicalizePass : Pass<"xtennn-canonicalize"> {
  let summary = "Perform XTenNN specific canonicalizations on the IR";
  let description = [{
    Pass that performs XTenNN specific foldings & canonicalizations to bring the
    IR into a common state.
  }];

  let constructor = "amd::xten_nn::createCanonicalizePass()";

  let options = [
    Option<"allowDestructivePatterns", "allow-destructive", "bool", /*default=*/"false",
           "Whether or not destructive, i.e. non-semantic preserving, patterns "
           "should be included.">,
  ];
}

#endif // XTENNN_PASSES