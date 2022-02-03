//===- XTenPasses.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_DIALECT_PASSES
#define XTEN_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def XTenName : Pass<"xten-name-layers", "ModuleOp"> {
  let summary = "Give unique name attributes to xten operations";
  let constructor = "xilinx::xten::createXTenNamePass()";
}

def XTenDataflow : Pass<"xten-expand-graph", "ModuleOp"> {
  let summary = "xten expand graph";
  let constructor = "xilinx::xten::createXTenDataflowPass()";
}

#endif
