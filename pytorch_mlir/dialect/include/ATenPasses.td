#ifndef ATEN_CONVERSION_PASSES
#define ATEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AcapLoopLowering : Pass<"aten-to-affine", "ModuleOp"> {
  let summary = "ATen dialect lowering to Affine Loops";
  let constructor = "xilinx::aten::createAcapLoopLoweringPass()";
}

#endif // ATEN_CONVERSION_PASSES
