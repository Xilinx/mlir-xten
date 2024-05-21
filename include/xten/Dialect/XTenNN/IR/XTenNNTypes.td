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

def XTenNN_AnyTensorSignlessOrUnsignedInteger : TensorOf<
    [AnySignlessInteger, AnyUnsignedInteger], 
    [], 
    "signless-or-unsigned-tensor">;

def XTenNN_AnyTensorFloat : TensorOf<
    [AnyFloat],
    [],
    "signless-or-unsigned-tensor">;

#endif // XTENNN_TYPES