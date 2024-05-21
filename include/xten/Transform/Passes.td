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

#ifndef ATEN_TRANSFORM_PASSES
#define ATEN_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

// def ATenLowering : Pass<"aten-lowering", "ModuleOp"> {
//   let summary = "ATen dialect lowering to function calls";
//   let constructor = "xilinx::xten::createATenLoweringPass()";
// }

def ATenOpReport : Pass<"aten-op-report", "ModuleOp"> {
  let summary = "Generate ATen dialect operation statistics";
  let constructor = "xilinx::xten::createATenOpReportPass()";
}

def LowerToLibATen : Pass<"lower-to-libaten", "ModuleOp"> {
  let summary = "Lower ATen Ops to library calls";
  let constructor = "xilinx::xten::createLowerToLibATenPass()";
}

def XTenMinimizeLiveTensors : Pass<"xten-minimize-live", "func::FuncOp"> {
  let summary = "Reorder xten operations to minimize the total size of the live tensors";
  let constructor = "xilinx::xten::createXTenMinimizeLiveTensorsPass()";
  let options = [
    Option<"printSchedCost", "dot-sched-cost", "bool", /*default=*/"false",
           "Prints a dot representation of the cost associated with operations "
           "of the function in stderr.">,
    Option<"printBeforeSched", "dot-before-sched", "bool", /*default=*/"false",
           "Prints a dot representation of the sequence of operations in the DAG "
           "in the order they are to be executed before scheduling in stderr.">,
    Option<"printAfterSched", "dot-after-sched", "bool", /*default=*/"false",
           "Prints a dot representation of the sequence of operations in the DAG "
           "in the order they are to be executed after scheduling is done in stderr.">,
  ];
}

#endif // ATEN_TRANSFORM_PASSES
