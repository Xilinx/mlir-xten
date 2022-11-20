//===- XTenNamePass.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "xten/Dialect/XTen/XTenNamePass.h"

#include <iostream>

#define DEBUG_TYPE "xten-name-pass"

using namespace mlir;
using namespace xilinx::xten;

namespace xilinx {
namespace xten {

struct XTenNamePass : public XTenNameBase<XTenNamePass> {
public:

  std::string getLayerName(std::string layerPrefix, uint64_t id) {
    return layerPrefix + std::to_string(id);
  }

  void runOnOperation() override {
    markAllAnalysesPreserved();

    ModuleOp module = getOperation();

    auto forward = module.lookupSymbol<func::FuncOp>("forward");
    if(!forward) {
      emitError(UnknownLoc::get(module.getContext()),
                "Can't find forward function\n");
      signalPassFailure();
      return;
    }

    std::map<std::string, uint64_t> layerToName;
    forward.walk([&](Operation *op) {
      llvm::StringRef opName = op->getName().getStringRef();

      if(!opName.startswith(llvm::StringRef("torch.aten.")) &&
         !opName.startswith(llvm::StringRef("xten."))) {
        return; // skips basicpy constant generation and similar
      }

      if(opName.startswith("xten.") && !opName.find("conv")) {
        return; // Only interested about actual layers of the NN
      }

      llvm::StringRef type;
      if (opName.startswith("torch.aten.")) {
        type = opName.split("torch.aten.").second;
      } else {
        type = opName.split("xten.").second;
      }

      if (type.equals("constant")) {
        return;
      }

      std::string layerName;
      if (layerToName.count(type.str())) {
        layerToName[type.str()] = layerToName[type.str()] + 1;
      } else {
        layerToName[type.str()] = 0;
      }

      layerName = getLayerName(type.str(), layerToName[type.str()]);
      auto attr = StringAttr::get(module.getContext(), layerName);
      if (op->hasAttr("layer_name") && !overwriteExisting) {
        return;
      }

      op->setAttr(llvm::StringRef("layer_name"), attr);
    });
  }
};

}
}

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createXTenNamePass() {
  return std::make_unique<XTenNamePass>();
}

} // namespace xten
} // namespace xilinx


