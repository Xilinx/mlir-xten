//===-- XTenNNBase.td - XTenNN dialect definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_BASE
#define XTENNN_BASE

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterface.td"

def XTenNN_Dialect : Dialect {
  let name = "xten_nn";
  let cppNamespace = "::amd::xten_nn";
  let summary = "The XTen Neural Network dialect";
  let description = [{
    The `xten_nn` dialect provides basic graph-like semantics. It implements a subgraph 
    operation that can wrap operations into regions and interfaces to handle those.
  }];

  code extraClassDeclaration = [{
    private:
        void registerOps();

    }];

  let useDefaultTypePrinterParser = 0;
}

#endif // XTENNN_BASE
