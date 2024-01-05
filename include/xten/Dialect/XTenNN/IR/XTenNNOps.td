//===-- XTenNNOps.td - XTenNN ops definitions *---------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_OPS
#define XTENNN_OPS

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"
include "xten/Dialect/XTenNN/IR/XTenNNTypes.td"

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
            DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
            SingleBlockImplicitTerminator<"OutputOp">,
            XTenNN_EnclaveOp,
            IsolatedFromAbove,
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


def XTenNN_QuantizeOp: XTenNN_Op<"quantize", [
            Elementwise,
            Pure,
            SameOperandsAndResultShape]> {
  let summary = "Quantizes a float32 tensor to a signless or unsigned integer tensor of given width.";
  let description = [{
    Quantizes a given float32 tensor into a signless or unsigned integer tensor of given width.
    Since tosa is using signless/unsigned types currently, we also consider signless integer types for
    signed ones when the type is not unsigned until tosa support signed integers.

    Applies the following linear quantization to the input tensor x:
      y = round( x / 2^shift )

    Where 2^shift is equal to the scale of the quantize operation and
    the shift is an attribute of the operation in si32.

    Round will saturate to the range of the output type and the rounding mode is set to half 
    to nearest even.
  }];

  let arguments = (ins 
    F32Tensor:$input,
    SI32Attr:$shift
  );

  let results = (outs XTenNN_AnyTensorSignlessOrUnsignedInteger:$output);

  let assemblyFormat = [{ `(`$input `:` type($input)`)` attr-dict `->` type($output) }];

  let hasFolder = 1;
}

def XTenNN_DequantizeOp: XTenNN_Op<"dequantize", [
            Elementwise,
            Pure,
            SameOperandsAndResultShape]> {
  let summary = "Dequantizes a signless/unsigned integer tensor of given bitwidth to a float32 tensor.";
  let description = [{
    Dequantizes a signless/unsigned integer tensor of given bitwidth to a float32 tensor.
    Since tosa is using signless/unsigned types currently, we also consider signless integer types for
    signed ones when the type is not unsigned until tosa support signed integers.

    Applies the following linear dequantization to the input tensor x:
      y = x  * ( 2^shift )

    Where 2^shift is equal to scale of the dequantize operation and
    the shift is an attribute of the operation in si32.
  }];

  let arguments = (ins 
    XTenNN_AnyTensorSignlessOrUnsignedInteger:$input,
    SI32Attr:$shift
  );

  let results = (outs F32Tensor:$output);

  let assemblyFormat = [{ `(`$input `:` type($input)`)` attr-dict `->` type($output) }];
}

def XTenNN_LoadExternalConstOp: XTenNN_Op<"load_external_const", [
            Pure]> {
  let summary = "Loads a constant from an external h5 file to a const operator.";
  let description = [{
    Looks into `file` for `key`, retrieves the value and replace this operator by the loaded value.

    Unfortunately, this operation cannot carry the ConstantLike trait as a Fold operation
    is required to be implemented for constants. For this particular operation we cannot return
    a sensible value as other dialect constants do since the value is stored within a file.

    This implementation follows closely to what `ml_program.global_load_const` implements. As it too
    does not implement the ConstantLike trait.
  }];
  let arguments = (ins 
    StrAttr:$key,
    StrAttr:$file
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{ attr-dict `->` type($output) }];
}

//===----------------------------------------------------------------------===//
// Ops that are missing from the TOSA standard
//===----------------------------------------------------------------------===//

def TosaExtension : NativeOpTrait<"TosaExtension">;
def ElementwiseUnary : NativeOpTrait<"ElementwiseUnary">;

def XtenNN_SignOp: XTenNN_Op<"sign", [Pure, TosaExtension, ElementwiseUnary, SameOperandsAndResultElementType]> {
  let summary = "Calculate the sign of the given input tensor element-wise";
  let description = [{
    Calculate the sign of the given input tensor element-wise. If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
    If the input is a NaN, the output will be a copy of the input element.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs
    AnyTensor:$output
  );

  let assemblyFormat = [{ operands attr-dict `:` functional-type(operands, results) }];
}

def XtenNN_MishOp: XTenNN_Op<"mish", [Pure, TosaExtension, ElementwiseUnary, SameOperandsAndResultElementType]> {
  let summary = "Calculate the mish operation of the given input tensor element-wise";
  let description = [{
    Calculate the mish operation (Self Regularized Non-Monotonic Neural Activation Function) of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs
    AnyTensor:$output
  );

  let assemblyFormat = [{ operands attr-dict `:` functional-type(operands, results) }];
}

#endif // XTENNN_OPS
