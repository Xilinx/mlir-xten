#ifndef XTEN_CONVERSION_PASSES
#define XTEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ATenToXTen : Pass<"aten-to-xten", "ModuleOp"> {
  let summary = "Convert ATen dialect to XTen dialect";
  let constructor = "xilinx::xten::createATenToXTenPass()";
}

def XTenToAffine : Pass<"xten-to-affine", "ModuleOp"> {
  let summary = "Convert XTen dialect to Affine dialect";
  let constructor = "xilinx::xten::createXTenToAffinePass()";
}

def XTenToLinalg : Pass<"xten-to-linalg", "ModuleOp"> {
  let summary = "Convert XTen dialect to Linalg dialect";
  let constructor = "xilinx::xten::createXTenToLinalgPass()";
}

#endif
