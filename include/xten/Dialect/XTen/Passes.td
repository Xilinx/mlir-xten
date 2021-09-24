#ifndef XTEN_DIALECT_PASSES
#define XTEN_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def XTenName : Pass<"xten-name-layers", "ModuleOp"> {
  let summary = "Give unique name attributes to xten operations";
  let constructor = "xilinx::xten::createXTenNamePass()";
}

def XTenDataFlow : Pass<"xten-expand-graph", "ModuleOp"> {
  let summary = "xten expand graph";
  let constructor = "xilinx::xten::createXTenDataflowPass()";
}

#endif
