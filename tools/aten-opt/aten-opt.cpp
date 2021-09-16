// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

//#include "mlir/Analysis/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"

#include "aten/Transform/Passes.h"
#include "aten/Conversion/Passes.h"
#include "aten/Dialect/XTen/XTenDialect.h"
#include "aten/Dialect/XTen/XTenPasses.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  xilinx::aten::registerTransformPasses();
  xilinx::aten::registerConversionPasses();
  xilinx::xten::registerXTenPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<xilinx::xten::XTenDialect,
                  NPCOMP::aten::ATenDialect,
                  NPCOMP::Basicpy::BasicpyDialect>();

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
