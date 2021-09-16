#ifndef ATEN_CONVERSION_PASSES
#define ATEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ATenLowering : Pass<"aten-lowering", "ModuleOp"> {
  let summary = "ATen dialect lowering to function calls";
  let constructor = "xilinx::aten::createATenLoweringPass()";
}

def ATenOpReport : Pass<"aten-op-report", "ModuleOp"> {
  let summary = "Generate ATen dialect operation statistics";
  let constructor = "xilinx::aten::createATenOpReportPass()";
}

def LowerToLibATen : Pass<"lower-to-libaten", "ModuleOp"> {
  let summary = "Lower ATen Ops to library calls";
  let constructor = "xilinx::aten::createLowerToLibATenPass()";
}

#endif // ATEN_CONVERSION_PASSES
