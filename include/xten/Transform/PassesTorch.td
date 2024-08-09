//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef ATEN_TRANSFORM_PASSES_TORCH
#define ATEN_TRANSFORM_PASSES_TORCH

include "mlir/Pass/PassBase.td"

//def ATenLowering : Pass<"aten-lowering", "ModuleOp"> {
//  let summary = "ATen dialect lowering to function calls";
//  let constructor = "xilinx::xten::createATenLoweringPass()";
//}

def ATenOpReport : Pass<"aten-op-report", "ModuleOp"> {
  let summary = "Generate ATen dialect operation statistics";
  let constructor = "xilinx::xten::createATenOpReportPass()";
}

def LowerToLibATen : Pass<"lower-to-libaten", "ModuleOp"> {
  let summary = "Lower ATen Ops to library calls";
  let constructor = "xilinx::xten::createLowerToLibATenPass()";
}

#endif // ATEN_TRANSFORM_PASSES_TORCH
