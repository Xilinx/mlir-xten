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

def XTenNNDictLocation : XTenNN_LocAttr<"XTenNNDictLocation"> {
  let mnemonic = "dict_loc";
  let parameters = (ins "mlir::DictionaryAttr":$attrs);

  let assemblyFormat = "`(` $attrs `)`";
}

def UnspecifiedAttr : AttrDef<XTenNN_Dialect, "Unspecified", []> {
  let mnemonic = "unspecified";
  let summary = "Marker for an unspecified value";
  let description = [{
    The `#xten_nn.unspecified` attribute marks a value as unspecified.

    The attribute itself is untyped. Its semantics come from the typed context
    in which it is used:
    - Integer-typed contexts allow any integer value.
    - Floating-point contexts allow any finite, non-NaN, non-infinite value.
  }];
}


#endif // XTENNN_TYPES