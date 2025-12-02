//===-- XTenNNTypes.td - XTenNN Type definitions *-------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_TYPES
#define XTENNN_TYPES

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"
include "mlir/IR/AttrTypeBase.td"

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

def XtenNN_TypeHasAxis : NativeOpTrait<"HasAxisTrait"> {
  let cppNamespace = XTenNN_Dialect.cppNamespace;
}

class XtenNN_Type<string name, list<Trait> traits = []> : TypeDef<XTenNN_Dialect, name, /*traits*/ traits > {
  let mnemonic = name;
}

class XtennNN_PerAxisQuantizedType<string name> :
    XtenNN_Type<name, [XtenNN_TypeHasAxis]> {
  let parameters = (ins "uint64_t":$axis);
  let assemblyFormat = "`<``axis``=` $axis `>`";
}

def XtenNN_MicroExponent6 : XtennNN_PerAxisQuantizedType<"MX6">;

def XtenNN_AnyPerAxisQuantizedType : AnyTypeOf<[XtenNN_MicroExponent6]>;

def IsAxisInRange : CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getElementType()">;

def XtenNN_PerAxisQuantizedTensor : RankedTensorOf<[XtenNN_AnyPerAxisQuantizedType], [IsAxisInRange]>;





def XTenNN_AnyIntegerOrFloat : AnyTypeOf<[AnyInteger, AnyFloat, XtenNN_PerAxisQuantizedTensor], "Integer or Float">;


#endif // XTENNN_TYPES