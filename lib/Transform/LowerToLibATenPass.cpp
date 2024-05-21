//===- LowerToLibATenPass.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

// This pass demangles function calls to match libaten_ops

#include "PassDetail.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <vector>

#define DEBUG_TYPE "lower-to-libaten"

using namespace mlir;
using namespace xilinx::xten;

namespace {

void CpuLibDemangle(ModuleOp op)
{
  std::map<std::string, std::string> nameMap;
  for (auto function : op.getOps<mlir::LLVM::LLVMFuncOp>()) {

    if (!function.isExternal())
      continue;
    if (!function.getName().startswith("_mlir_ciface_"))
      continue;
    std::string old_name = function.getName().str();
    auto extern_name = function.getName().split("_mlir_ciface_").second;
    auto name_mangling = extern_name.split("_AtenAcapOp_");
    std::string name = name_mangling.first.str();
    std::string mangling = name_mangling.second.str();
    std::stringstream new_mangling;

    for (int i=0, e=mangling.size(); i<e;) {
      char c = mangling[i++];
      if (c == 'M') {
        c = mangling[i++];
        int dim = 0;
        while (i != e && c != 'I' && c != 'F') {
          c = mangling[i++];
          if (c == 'x') dim++;
        }
        new_mangling << dim << c;
        c = mangling[i];
        while (i != e && c != '_') {
          new_mangling << c;
          c = mangling[++i];
        }
        new_mangling << '_';
      }
      // the functions have the suffix 'out'
      if (c == 'o' || c == 'u' || c == 't') {
        new_mangling << c;
      }
    }

    std::string new_name = new_mangling.str();
    // if no tensor parameters, pass it through
    if (new_name.size() == 0)
      new_name = name;
    // remove trailing underscore
    else if (new_name[new_name.size()-1] == '_')
      new_name = name + "_" + new_name.substr(0, new_name.size()-1);
    else
      new_name = name + "_" + new_name;

    nameMap[old_name] = new_name;
    while (op.lookupSymbol(new_name))
      new_name = new_name + "_";
    function.setName(new_name);
  }

  // update callsites
  for (auto function : op.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (function.isExternal())
      continue;
    function.walk([&](Operation *op) {
      if (auto callOp = dyn_cast<mlir::LLVM::CallOp>(op)) {
        auto attr = callOp->getAttrOfType<FlatSymbolRefAttr>("callee");
        if (nameMap.count(attr.getValue().str()) == 0)
          return;
        auto fn = nameMap[attr.getValue().str()];
        callOp->setAttr("callee", mlir::SymbolRefAttr::get(op->getContext(), fn));
      }
    });
  }
}

//#include "LowerToLibATen.cpp.inc"

class LowerToLibATenPass : public LowerToLibATenBase<LowerToLibATenPass> {

public:
  LowerToLibATenPass() = default;
  LowerToLibATenPass(const LowerToLibATenPass &pass){};

  void runOnOperation() override {
    CpuLibDemangle(getOperation());
  }

private:

};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerToLibATenPass() {
  return std::make_unique<LowerToLibATenPass>();
}

} // namespace xten
} // namespace xilinx

