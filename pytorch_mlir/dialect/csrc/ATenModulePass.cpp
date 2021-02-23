// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

// This is an empty module pass.  Copy it to make your own pass.

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/Pass.h"

#include <vector>

#define DEBUG_TYPE "aten-module-pass"

using namespace mlir;

namespace {

class ATenModulePass : public PassWrapper<ATenModulePass,
                                          OperationPass<ModuleOp>> {

public:
  ATenModulePass() {}

  void runOnOperation() override {

    auto module = getOperation();

    // check that a function called "graph" exists
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    graph.walk([&](Operation *op) {
    });

  }

private:

};

} // namespace

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenModulePass() {
  return std::make_unique<ATenModulePass>();
}

} // namespace aten
} // namespace xilinx
