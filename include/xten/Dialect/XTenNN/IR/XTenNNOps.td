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
            SingleBlockImplicitTerminator<"OutputOp">,
            XTenNN_EnclaveOp,
            IsolatedFromAbove,
            OpAsmOpInterface,
            RecursivelySpeculatable,
            RecursiveMemoryEffects]> {
    let summary = "Separates a subgraph inside a graph";
    let description = [{
        The `xten_nn.subgraph` operation declares its body to be an isolated sub-
        graph, separated from the surrounding graph.

        This allows code motion between the parent and anonymous
        subgraphs.

        Example:
        ```mlir
        func.func @subgraph(%arg0:  tensor<2xi64>) ->  tensor<2xi64> {
            %sum = xten_nn.subgraph (%c0 = %arg0 :  tensor<2xi64>) {
                // Implementation...
                xten_nn.output %out :  tensor<120xi64>
            } -> tensor<120xi64>
        return %sum :  tensor<120xi64>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$captures);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;


    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() { return "xten_nn"; }
    }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def XTenNN_OutputOp : XTenNN_Op<"output", [
            HasParent<"SubgraphOp">,
            Pure,
            Terminator,
            ReturnLike]> {
    let summary = "Defines the output value of a subgraph or node";
    let description = [{
        The `xten_nn.output` operation serves as the terminator for XTenNN operations
        that declare a region that produces result values.

        Example:
        ```mlir
        %sum = xten_nn.subgraph (%c0 = %arg0 :  tensor<2xi64>) {
            // Implementation...
            xten_nn.output %result : tensor<120xi64>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = [{ attr-dict ($operands^ `:` type($operands))?}];
}


#endif // XTENNN_OPS