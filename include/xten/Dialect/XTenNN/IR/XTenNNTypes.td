//===-- XTenNNTypes.td - XTenNN Type definitions *-------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_TYPES
#define XTENNN_TYPES

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"

def XTenNN_AnyTensorSignedOrUnsignedInteger : TensorOf<
    [AnySignedInteger, AnyUnsignedInteger], 
    [], 
    "signed-or-unsigned-tensor">;

def XTeNN_F32AttrPowerOfTwo : TypedAttrBase<F32, "FloatAttr",
              And<[CPred<"$_self.isa<::mlir::FloatAttr>()">,
                   CPred<"$_self.cast<::mlir::FloatAttr>().getType().isF32()">, 
                   CPred<"amd::xten_nn::isPowerOfTwoFloat($_self)">]>,
              "float32 is a Power-Of-Two value"> {
  let returnType = [{ ::llvm::APFloat }];
}

#endif // XTENNN_TYPES