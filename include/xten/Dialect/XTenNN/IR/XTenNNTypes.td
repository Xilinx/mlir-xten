//===-- XTenNNTypes.td - XTenNN Type definitions *-------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_TYPES
#define XTENNN_TYPES

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"

def XTenNN_AnySignlessOrUnsignedIntegerTensor : TensorOf<
    [AnySignlessInteger, AnyUnsignedInteger], 
    [], 
    "signless-or-unsigned-tensor">;

def XTenNN_AnyFloatTensor : TensorOf<[AnyFloat]>;

def XTenNN_AnyIntegerAttr:
    TypedAttrBase<
      AnyInteger, "IntegerAttr",
      CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
      "Any integer attr"> {
  let returnType = [{ ::llvm::APInt }];
  let constBuilderCall = ?;
}

def XTenNN_AnyIntegerOrFloat : AnyTypeOf<[AnyInteger, AnyFloat], "Integer or Float">;

class XTenNN_LocAttr<string name> : LocationAttrDef<XTenNN_Dialect, name, []>;

def XTenNNSubgraphLocation : XTenNN_LocAttr<"XTenNNSubgraphLocation"> {
  let mnemonic = "subgraph_loc";
  let parameters = (ins "mlir::DictionaryAttr":$attrs);

  let assemblyFormat = "`(` $attrs `)`";
}


#endif // XTENNN_TYPES