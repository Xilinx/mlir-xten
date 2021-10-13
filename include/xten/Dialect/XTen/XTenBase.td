//===- XTenBase.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_BASE
#define XTEN_BASE

include "mlir/IR/OpBase.td"

def XTen_Dialect : Dialect {
  let name = "xten";
  let cppNamespace = "xilinx::xten";
  let description = [{
The `xten` dialect is an IR...
  }];
}

def XTen_AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def XTen_AnyScalar : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;

def XTen_AnyOptionalTensor : AnyTypeOf<[
  AnyTensor//,
//  Basicpy_NoneType
], "tensor or None">;

#endif // #ifndef XTEN_BASE
