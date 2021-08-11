#ifndef ATEN_CONVERSION_PASSES
#define ATEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LowerToLibATen : Pass<"lower-to-libaten", "ModuleOp"> {
  let summary = "Lower ATen Ops to library calls";
  let constructor = "xilinx::aten::createLowerToLibATenPass()";
}

#endif // ATEN_CONVERSION_PASSES
