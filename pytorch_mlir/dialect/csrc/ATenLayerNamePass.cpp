// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "ATenDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/Pass.h"

#include <iostream>
#include <vector>

#define DEBUG_TYPE "aten-layer-name"

using namespace mlir;

namespace {

struct ATenLayerNamePass : public ModulePass<ATenLayerNamePass> {

private:
  std::map<Operation *, std::string> opToName;

public:
  ATenLayerNamePass() {}

  void runOnModule() override {

    markAllAnalysesPreserved();

    auto module = getModule();

    // find the function called 'graph'
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    // Construct a name for each aten operation
    std::map<std::string, uint64_t> layerIDmap;
    unsigned currentLayer = 0;

    graph.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();

      // if it's not an aten operation, continue
      if (!name.startswith("aten.")) return;

      // strip the aten prefix to get the operation type
      auto type = name.split("aten.").second;

      // if it's an aten constant op, continue
      if (type.equals("constant")) return;

      unsigned ID = 0;
      if (layerIDmap.count(type) == 0)
        layerIDmap[type] = 0;
      else
        ID = ++layerIDmap[type];

      std::string layerName = "L" + std::to_string(currentLayer++) + \
                              "-" + type.str() + "-" + std::to_string(ID);

      LLVM_DEBUG(llvm::dbgs() << "generated acdc_layer_name: '" << layerName << "'\n");

      auto attr = StringAttr::get(layerName, module.getContext());
      op->setAttr(StringRef("acdc_layer_name"), attr);

    });
  }
};

} // namespace

namespace xilinx {
namespace reports {

std::unique_ptr<mlir::Pass> createATenLayerNamePass() {
  return std::make_unique<ATenLayerNamePass>();
}

} // namespace reports
} // namespace xilinx