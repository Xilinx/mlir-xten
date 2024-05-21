//===- ATenOpReport.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2019 - 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "xten/Transform/ATenOpReport.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"


#include <iostream>
#include <vector>

#define DEBUG_TYPE "aten-op-stats"

using namespace mlir;
using namespace xilinx::xten;

namespace {
struct ATenOpReportPass : public ATenOpReportBase<ATenOpReportPass> {

private:
  std::string o;
  std::string &output;
  std::vector<std::string> tableFields;
  std::map<Operation *, std::string> opToName;

public:
  Option<std::string> ATenOpReportFilename{
      *this, "output-file", llvm::cl::desc("Output filename for JSON report"),
      llvm::cl::init("-")};

  ATenOpReportPass(const ATenOpReportPass &pass) : output(o) {}

  ATenOpReportPass()
      : output(o),
        tableFields({"reads", "writes", "activation_in", "activation_out",
                     "parameters_in", "ops:MAC", "ops:==", "ops:>", "ops:*",
                     "ops:+", "ops:/", "ops:sqrt", "ops:-", "grad"}) {}

  std::string emitJSONReport() {

    llvm::json::Object top;

    auto forward = getOperation().lookupSymbol<func::FuncOp>("forward");
    forward.walk([&](Operation *op) {
      std::map<std::string, uint64_t> layerStatsMap;
      layerStatsMap = xilinx::xten::getATenOpStats(op);
      if (!layerStatsMap.size())
        return;

      // name for this layer
      std::string layerName = opToName[op];

      // raw stats for this layer
      // JSON version of the stats we are building
      llvm::json::Object layerStatsJSON;

      // foreach string f in tableField,
      // get the sum of all entries in layerStatsMap containing f
      for (auto &f : tableFields) {
        for (auto &p : layerStatsMap) {
          if (p.first.find(f) != std::string::npos) {
            if (auto count = layerStatsJSON[f].getAsInteger())
              layerStatsJSON[f] = (int64_t)p.second + *count;
            else
              layerStatsJSON[f] = (int64_t)p.second;
          }
        }
      }
      top[layerName] = llvm::json::Value(std::move(layerStatsJSON));
    });

    llvm::json::Value topv(std::move(top));
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << llvm::formatv("{0:2}", topv) << "\n";
    return ss.str();
  }

  void runOnOperation() override {

    // I don't change anything
    markAllAnalysesPreserved();

    auto module = getOperation();

    // check that a function called "forward" exists
    auto forward = module.lookupSymbol<func::FuncOp>("forward");
    if (!forward) {
      emitError(UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a forward function\n");
      signalPassFailure();
      return;
    }

    unsigned currentLayer = 0;
    opToName.clear();
    forward.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("layer_name");
      if (attr)
        opToName[op] = attr.getValue().str();
      else
        opToName[op] = "unknown-layer-" + std::to_string(currentLayer);
      currentLayer++;
    });

    output = emitJSONReport();

    if (ATenOpReportFilename != "-") {
      std::error_code EC;
      llvm::raw_fd_ostream aie_ostream(ATenOpReportFilename, EC);
      aie_ostream << output;
    } else {
      llvm::outs() << output;
    }
  }
};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createATenOpReportPass() {
  return std::make_unique<ATenOpReportPass>();
}

} // namespace xten
} // namespace xilinx
