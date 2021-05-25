#ifndef ATEN_CONVERSION_PASSES
#define ATEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AcapLoopLowering : Pass<"aten-to-affine", "ModuleOp"> {
  let summary = "ATen dialect lowering to Affine Loops";
  let constructor = "xilinx::aten::createAcapLoopLoweringPass()";
}

def ATenToAIR : Pass<"aten-to-air", "ModuleOp"> {
  let summary = "ATen dialect lowering to AIR operations";
  let constructor = "xilinx::aten::createATenToAIRPass()";
}

def LowerToLibATen : Pass<"lower-to-libaten", "ModuleOp"> {
  let summary = "Lower ATen Ops to library calls";
  let constructor = "xilinx::aten::createLowerToLibATenPass()";
}

#endif // ATEN_CONVERSION_PASSES
