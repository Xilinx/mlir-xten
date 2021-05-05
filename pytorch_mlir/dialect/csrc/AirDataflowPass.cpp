// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AirDataflow.h"
#include "AIRDialect.h"
#include "AirDataflowUtils.h"
//#include "AirOpWrapper.h"

#include <iostream>
#include <vector>
#include <set>

#define DEBUG_TYPE "air-dataflow-pass"

#define AInAttr "AIn"
#define AOutAttr "AOut"

using namespace mlir;

// TODO make the patterns generic
//   - Restriction from architectural constraints
//   - Support all convolution types

// TODO Make sure that the communication is not shuffled somehow

// General idea: because it is easy to know from the analytical model what do we do with everything:
// That analytical model part is responsible for finding an optimal solution, and then it communicates it here
// Hence in that class we only try to make sure it is easy to generate a graph for any given input

namespace xilinx {
    namespace air {

        struct AirDataflowPass : public PassWrapper<AirDataflowPass, OperationPass<ModuleOp>> {
        private:
            std::map<std::string, std::vector<AbsOpWrapper*>> layerNameToOps;
            std::map<std::string, ModelParams> layerNameToParams;
            // TODO add some sync between structs and graph modifications

        public:
            AirDataflowPass() {}

            AbsOpWrapper* opToWrapper(Operation* op) {
                if(auto conv = llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    return new Conv2dReLUOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dReLUOp>(op)) {
                    return new PartialConv2dReLUOpWrapper(conv);
                } else if(auto maxpool = llvm::dyn_cast<mlir::NPCOMP::aten::MaxPool2dWithIndicesOp>(op)) {
                    return new MaxPool2dWithIndicesOpWrapper(maxpool);
                } else if(llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    llvm::outs() << "Unimplemented errror\n";
                    exit(1);
                } else if(llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    llvm::outs() << "Unimplemented errror\n";
                    exit(1);
                } else if(llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    llvm::outs() << "Unimplemented errror\n";
                    exit(1);
                }
            }

            void initializeLayerNameToOps(FuncOp graph) {
                // Fill layerNameToOps with basic information
                graph.walk([&](Operation *op) {
                        if(op->getAttr("name") != nullptr) {
                            auto opName = (op->getAttr("name").dyn_cast<StringAttr>()).getValue();
                            AbsOpWrapper* wrappedOp = opToWrapper(op);
                            if(layerNameToOps.count(opName.str()) == 0) {
                                layerNameToOps[opName.str()] = std::vector<AbsOpWrapper*>({wrappedOp});
                            } else {// TODO should never be reached
                                layerNameToOps[opName.str()].push_back(wrappedOp);
                            }
                        }
                    });
            }

            void clearLayerNameToOps() {
                std::map<std::string, std::vector<AbsOpWrapper*>>::iterator it;
                for(it = layerNameToOps.begin(); it != layerNameToOps.end(); it++) {
                    while(!it->second.empty()) {
                        delete it->second.back();
                        it->second.pop_back();
                    }
                }
            }

            void initializeLayerNameToParams(FuncOp graph) {
                if(this->layerNameToOps.size() == 0) {
                    initializeLayerNameToOps(graph);
                }

                std::map<std::string, std::vector<AbsOpWrapper*>>::iterator it;
                for(it = layerNameToOps.begin(); it != layerNameToOps.end(); it++) {
                    this->layerNameToParams[it->first] = ModelParams();
                }
            }

            // TODO how to make that generic with respect to Conv Ops? As much as possible?
            LogicalResult PTransform(std::string layerName, unsigned int into) {
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> cstsToDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    //Conv2dReLUOp op = llvm::dyn_cast<Conv2dReLUOp>(genOp);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = genOp->getWeights().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, PSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                    Operation* biases = genOp->getBiases().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, PSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split Return Type shape
                    ShapedType nReturnType = breakShapeInto(op->getResult(0).getType().dyn_cast<ShapedType>(), 1, into);
                    llvm::outs() << "Return Type: " << op->getResult(0).getType() << " and new is " << nReturnType << "\n";

                    // Generate new convs
                    for(unsigned int i = 0; i < into; i++) {
                        Operation* conv = genOp->buildOp(builder,
                                                        TypeRange({nReturnType}),
                                                        genOp->getInput(),
                                                        llvm::Optional<Value>(nConsts.at(i)),
                                                        llvm::Optional<Value>(nBiases.at(i)),
                                                        llvm::Optional<Value>(), false);

                        nConvs.push_back(conv->getResult(0));
                        nLayerOps.push_back(opToWrapper(conv));
                    }

                    // TODO make sure that dim of split is correct one
                    // if split afterwards check size of concat else concat
                    if(op->hasOneUse() && (llvm::dyn_cast<SplitOp>(*(op->getUsers().begin())))) {
                        SplitOp split = llvm::dyn_cast<SplitOp>(*(op->getUsers().begin()));
                        replaceSplit(builder, split, nConvs, cstsToDelete, COUT_LOC);
                    } else {
                        insertConcat(builder, op->getResult(0), nConvs, COUT_LOC);
                    }

                    // Delete previous Csts and ConvolutionOp
                    cstsToDelete.push_back(weights);
                    cstsToDelete.push_back(biases);
                }

                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(cstsToDelete);
                deleteOpsFrom(layerOps);


                return success();
            }

            LogicalResult CaTransform(std::string layerName, unsigned int into) {
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nInputs;
                    //std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = genOp->getWeights().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, CaSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                   Operation* biases = genOp->getBiases().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, CaSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // TODO not so beautiful but potentially check here if we also have fused BN params??

                    // split activations
                    if(auto constOp = genOp->getInput().getDefiningOp<ConstantOp>()) {
                        splitConstantInto(constOp, nInputs, builder, CaSplit, aSplitType, into);
                    } else {
                        if(ConcatOp concatOp = genOp->getInput().getDefiningOp<ConcatOp>()) {
                            replaceConcat(builder, concatOp, nInputs, toDelete, C_LOC, into);
                        } else {
                            insertSplit(builder, genOp->getInput(), nInputs, C_LOC, into);
                        }
                    }

                    // Generate convolutions
                    Operation* conv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                    nInputs.at(0), llvm::Optional<Value>(nConsts.at(0)),
                                                    llvm::Optional<Value>(nBiases.at(0)), llvm::Optional<Value>(), true);

                    nLayerOps.push_back(opToWrapper(conv));

                    for(unsigned int i = 1; i < into; i++) {
                        Operation* nConv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                         nInputs.at(i), llvm::Optional<Value>(nConsts.at(i)),
                                                         llvm::Optional<Value>(nBiases.at(i)),
                                                         llvm::Optional<Value>(conv->getResult(0)), true);

                        conv = nConv;
                        nLayerOps.push_back(opToWrapper(conv));
                    }

                    // Insert it in the graph
                    op->getResult(0).replaceAllUsesWith(conv->getResult(0));

                    // Prepare to delete
                    toDelete.push_back(weights);
                    toDelete.push_back(biases);
                }

                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(toDelete);
                deleteOpsFrom(layerOps);

                return success();
            }

            LogicalResult LTransform(std::string layerName, unsigned int into) {
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = genOp->getWeights().getDefiningOp();//->getName();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, LSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                    Operation* biases = genOp->getBiases().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, LSplit, bSplitType, 2);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Same return type here
                    ShapedType returnType = op->getResult(0).getType().dyn_cast<ShapedType>();

                    // Generate new convs
                    Operation* nConv = genOp->buildOp(builder, TypeRange({returnType, returnType}),
                                                     genOp->getInput(), llvm::Optional<Value>(nConsts.at(0)),
                                                     llvm::Optional<Value>(nBiases.at(0)), llvm::Optional<Value>(), true);

                    Value forward = nConv->getResult(1);
                    Value partial = nConv->getResult(0);

                    for(unsigned int i = 1; i < into; i++) {
                        genOp->buildOp(builder, (i == (into-1)) ? TypeRange({returnType}) : TypeRange({returnType, returnType}),
                                      forward, llvm::Optional<Value>(nConsts.at(i)),
                                      llvm::Optional<Value>(nBiases.at(i)),
                                      llvm::Optional<Value>(partial), false);

                        partial = nConv->getResult(0);
                        if(i < (into-1)) {
                            forward = nConv->getResult(1);
                        }

                    }


                    // Replace output of old convolution usage by concat value
                    op->getResult(0).replaceAllUsesWith(nConv->getResult(0));

                    toDelete.push_back(weights);
                    toDelete.push_back(biases);
                }

                // Delete previous Csts and ConvolutionOp
                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(toDelete);
                deleteOpsFrom(layerOps);

                return success();

            }

            // Allocate necessary attributes for tiling where line counts from 0 to max in flight at that layer
            // - AIn = ArrayAttr [min line, max line] -> can now bank number from this
            // - AOut = ArrayAttr [min line, max line] -> can now bank number from this
            void allocateTile(FuncOp graph) {
                // TODO
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<FuncOp>("graph");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Cant find graph func\n");
                    signalPassFailure();
                    return;
                }

                initializeLayerNameToOps(graph);
                initializeLayerNameToParams(graph);

                // expand slowest layer
                PTransform("conv2d_relu1", 4);
                CaTransform("conv2d_relu1", 4);

                clearLayerNameToOps();
            }
        };
    }
}

namespace xilinx {
    namespace air {
        std::unique_ptr<mlir::Pass> createAirDataflowPass() {
            return std::make_unique<AirDataflowPass>();
        }

    } // namespace aten
} // namespace xilinx

void xilinx::air::registerAirDataflowPass() {
    PassRegistration<AirDataflowPass>("air-expand-graph",
                                      "Dataflow expansion of ATen NN graph towards AIE implementation");
}

