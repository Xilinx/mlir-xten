//===-- XTenNNOps.td - XTenNN ops definitions *---------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_OPS
#define XTENNN_OPS

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class XTenNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<XTenNN_Dialect, mnemonic, traits> {
}


//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

def XTenNN_SubgraphOp : XTenNN_Op<"subgraph", [
            // HasOnlyGraphRegion,
            // RegionKindInterface,
            // SingleBlockImplicitTerminator<"OutputOp">,
            IsolatedFromAbove,
            OpAsmOpInterface,
            RecursiveSideEffects]> {
    let summary = "Separates a subgraph inside a graph";
    let description = [{
        The `xten_nn.subgraph` operation declares its body to be an isolated sub-
        graph, separated from the surrounding graph.

        This is equivalent to exporting that subgraph to an outlined
        `dlnn.graph` and then embedding it using `dlnn.embed`, but happens in-
        place. However, this allows code motion between the parent and anonymous
        subgraphs.

        Example:
        ```mlir
        dlnn.graph @graph1(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            %result = dlnn.subgraph (%cap0 = %arg0 : !fm<3xf16[108,108]>) {
                // Implementation...
                dlnn.output %rel0 : !fm<128xf16>
            } -> !fm<128xf16>
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$captures);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() { return "xten_nn"; }
    }];
}

#endif // XTENNN_OPS