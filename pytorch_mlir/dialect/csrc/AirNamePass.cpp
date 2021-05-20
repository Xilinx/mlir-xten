// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AirName.h"

#include <iostream>

#define DEBUG_TYPE "air-name-pass"

using namespace mlir;

// heavily inspired from naming pass in ATenNamePass

namespace xilinx {
    namespace air {
        struct AIRNamePass : public PassWrapper<AIRNamePass, OperationPass<ModuleOp>> {
        public:
            std::string getLayerName(std::string layerPrefix, uint64_t id) {
                return layerPrefix + std::to_string(id);
            }

            void runOnOperation() override {
                markAllAnalysesPreserved();

                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<FuncOp>("graph");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Can't find graph function\n");
                    signalPassFailure();
                    return;
                }

                std::map<std::string, uint64_t> layerToName;
                graph.walk([&](Operation *op) {
                        llvm::StringRef opName = op->getName().getStringRef();

                        if(!opName.startswith(llvm::StringRef("aten.")) && !opName.startswith(llvm::StringRef("air."))) {
                            return; // skips basicpy constant generation and similar
                        }

                        if(opName.startswith("air.") && !opName.find("conv")) {
                            return; // Only interested about actual layers of the NN
                        }

                        llvm::StringRef type;
                        if(opName.startswith("aten.")) {
                            type = opName.split("aten.").second;
                        } else {
                            type = opName.split("air.").second;
                        }

                        if(type.equals("constant")) {
                            return;
                        }

                        std::string layerName;
                        if(layerToName.count(type.str())) {
                            layerToName[type.str()] = layerToName[type.str()] + 1;
                        } else {
                            layerToName[type.str()] = 0;
                        }

                        layerName = getLayerName(type.str(), layerToName[type.str()]);
                        auto attr = StringAttr::get(module.getContext(), layerName);
                        op->setAttr(llvm::StringRef("name"), attr);
                    });
            }
        };
    }
}

namespace xilinx {
    namespace air {
        std::unique_ptr<mlir::Pass> createAirNamePass() {
            return std::make_unique<AIRNamePass>();
        }

    } // namespace aten
} // namespace xilinx

void xilinx::air::registerAirNamePass() {
    PassRegistration<AIRNamePass>("air-name-layers",
                                  "Give a unique name to all compute layers of a NN");
}


