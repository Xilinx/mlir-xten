// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "ATenDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#include "mlir/Pass/Pass.h"

#include <iostream>
#include <vector>

#define DEBUG_TYPE "aten-op-report"

using namespace mlir;

namespace {

// Some helper functions
std::string getLayerName(std::string layer_type, std::map<std::string, uint64_t>  &layerIDmap, unsigned currentLayer) {

  unsigned ID = 0;
  if (layerIDmap.count(layer_type) == 0)
    layerIDmap[layer_type] = 0;
  else
    ID = ++layerIDmap[layer_type];

  return("L" + std::to_string(currentLayer) + "-" + layer_type + "-" + std::to_string(ID));
}

std::string getAsString(std::map<std::string, uint64_t> &m, std::string &e) {
  return m.count(e) ? std::to_string(m[e]) : " ";
}

struct ATenOpReportPass : public ModulePass<ATenOpReportPass> {

private:
  std::string &output;
  std::vector<std::string> tableFields;
  std::map<Operation *, std::string> opToName;

public:
  ATenOpReportPass(std::string &output)
    : output(output),
      tableFields({
        "reads",
        "writes",
        "activation_in",
        "activation_out",
        "parameters_in",
        "MAC",
        "==",
        ">",
        "*",
        "+",
        "/",
        "sqrt",
      })
  {
  }

  std::string emitJSONReport() {

    auto insertJSON = [&](llvm::json::Object &t,
                          std::string layerName,
                          std::map<std::string, uint64_t> &statMap) -> void {
      llvm::json::Object m;
      for(auto &f: tableFields) {
        if(statMap.count(f))
          m[f] = getAsString(statMap, f);
      }
      llvm::json::Value v(std::move(m));
      t[layerName] = v;
    };

    llvm::json::Object top;
    // Walk all the operations and a row for each one to the table.
    for (auto function : getModule().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        if (auto stats = mlir::dyn_cast<xilinx::aten::StatisticsOpInterface>(op)) {
          std::string layerName = opToName[op];
          std::map<std::string, uint64_t> u = stats.getStatistics();
          insertJSON(top, layerName, u);
        }
      });
    }
    llvm::json::Value topv(std::move(top));
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << llvm::formatv("{0:2}",topv) << "\n";
    return ss.str();
  }

  void runOnModule() override {

    // I don't change anything
    markAllAnalysesPreserved();

    auto module = getModule();

    // check that a function called "graph" exists
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    // Construct a name for each op.  This should be pulled into a seperate pass
    std::map<std::string, uint64_t> layerIDmap;
    unsigned currentLayer = 0;
    opToName.clear();

    for (auto function : module.getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
          auto name = op->getName().getStringRef();
          std::string layerName = getLayerName(name, layerIDmap, currentLayer++);
          opToName[op] = layerName;
      });
    }

    output = emitJSONReport();
  }
};

} // namespace

namespace xilinx {
namespace reports {

std::unique_ptr<mlir::Pass> createATenOpReportPass(std::string &o) {
  return std::make_unique<ATenOpReportPass>(o);
}

} // namespace reports
} // namespace xilinx