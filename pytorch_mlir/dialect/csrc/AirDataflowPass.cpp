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
//#include "AirDataflowExplorer.h"

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
            std::map<std::string, std::vector<Operation*>> layerNameToOps;
            std::map<std::string, ModelParams> layerNameToParams;
            // TODO add some sync between structs and graph modifications

        public:
            AirDataflowPass() {}

            void initializeLayerNameToOps(FuncOp graph) {
                // Fill layerNameToOps with basic information
                graph.walk([&](Operation *op) {
                        if(op->getAttr("name") != nullptr) {
                            auto opName = (op->getAttr("name").dyn_cast<StringAttr>()).getValue();
                            if(layerNameToOps.count(opName.str()) == 0) {
                                layerNameToOps[opName.str()] = std::vector<Operation*>({op});
                            } else {// TODO should never be reached
                                layerNameToOps[opName.str()].push_back(op);
                            }
                        }
                    });
            }

            void initializeLayerNameToParams(FuncOp graph) {
                if(this->layerNameToOps.size() == 0) {
                    initializeLayerNameToOps(graph);
                }

                std::map<std::string, std::vector<Operation*>>::iterator it;
                for(it = layerNameToOps.begin(); it != layerNameToOps.end(); it++) {
                    this->layerNameToParams[it->first] = ModelParams();
                }
            }

            // TODO how to make that generic with respect to Conv Ops? As much as possible?
            LogicalResult PTransform(std::string layerName, unsigned int into) {
                std::vector<Operation*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> cstsToDelete;
                std::vector<Operation*> nLayerOps;

                for(Operation* genOp : layerOps) {
                    OpBuilder builder(genOp);

                    Conv2dReLUOp op = llvm::dyn_cast<Conv2dReLUOp>(genOp);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = op.weight().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, PSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                    Operation* biases = op.bias().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, PSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split Return Type shape
                    ShapedType nReturnType = breakShapeInto(op.getResult().getType().dyn_cast<ShapedType>(), 1, into);
                    llvm::outs() << "Return Type: " << op.getResult().getType() << " and new is " << nReturnType << "\n";

                    // Generate new convs
                    for(unsigned int i = 0; i < into; i++) {
                        Operation* conv = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                                       nReturnType,
                                                                       op.input(),
                                                                       nConsts.at(i),
                                                                       nBiases.at(i),
                                                                       op.stride(),
                                                                       op.padding(),
                                                                       op.dilation(),
                                                                       op.transposed(),
                                                                       op.output_padding(),
                                                                       op.groups());

                        nConvs.push_back(conv->getResult(0));
                        nLayerOps.push_back(conv);
                    }

                    // TODO make sure that dim of split is correct one
                    // if split afterwards check size of concat else concat
                    if(op->hasOneUse() && (llvm::dyn_cast<SplitOp>(*(op->getUsers().begin())))) {
                        SplitOp split = llvm::dyn_cast<SplitOp>(*(op->getUsers().begin()));
                        replaceSplit(builder, split, nConvs, cstsToDelete, COUT_LOC);
                    } else {
                        insertConcat(builder, op.getResult(), nConvs, COUT_LOC);
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
                std::vector<Operation*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<Operation*> nLayerOps;

                for(Operation* genOp : layerOps) {
                    OpBuilder builder(genOp);

                    Conv2dReLUOp op = llvm::dyn_cast<Conv2dReLUOp>(genOp);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nInputs;
                    //std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = op.weight().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, CaSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                   Operation* biases = op.bias().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, CaSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // split activations
                    if(auto constOp = op.input().getDefiningOp<ConstantOp>()) {
                        splitConstantInto(constOp, nInputs, builder, CaSplit, aSplitType, into);
                    } else {
                        if(ConcatOp concatOp = op.input().getDefiningOp<ConcatOp>()) {
                            replaceConcat(builder, concatOp, nInputs, toDelete, C_LOC, into);
                        } else {
                            insertSplit(builder, op.input(), nInputs, C_LOC, into);
                        }
                    }

                    // Generate convolutions
                    Operation* conv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                          op.getResult().getType(),
                                                                          nInputs.at(0),
                                                                          nullptr,
                                                                          nConsts.at(0),
                                                                          nBiases.at(0),
                                                                          op.stride(),
                                                                          op.padding(),
                                                                          op.dilation(),
                                                                          op.transposed(),
                                                                          op.output_padding(),
                                                                          op.groups());
                    nLayerOps.push_back(conv);

                    for(unsigned int i = 1; i < into; i++) {
                        Operation* nConv =  builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                                op.getResult().getType(),
                                                                                nInputs.at(i),
                                                                                conv->getResult(0),
                                                                                nConsts.at(0),
                                                                                nBiases.at(0),
                                                                                op.stride(),
                                                                                op.padding(),
                                                                                op.dilation(),
                                                                                op.transposed(),
                                                                                op.output_padding(),
                                                                                op.groups());
                        conv = nConv;
                        nLayerOps.push_back(conv);
                    }

                    // Insert it in the graph
                    op.getResult().replaceAllUsesWith(conv->getResult(0));

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
                std::vector<Operation*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<Operation*> nLayerOps;

                for(Operation* genOp : layerOps) {
                    OpBuilder builder(genOp);

                    Conv2dReLUOp op = llvm::dyn_cast<Conv2dReLUOp>(genOp);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;

                    // Split weights
                    Operation* weights = op.weight().getDefiningOp();//->getName();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, LSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Split biases
                    Operation* biases = op.bias().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, LSplit, bSplitType, 2);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Same return type here
                    ShapedType returnType = op.getResult().getType().dyn_cast<ShapedType>();

                    // Generate new convs
                    Operation* nConv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                           TypeRange({returnType, returnType}),
                                                                           op.input(),
                                                                           nullptr,
                                                                           nConsts.at(0),
                                                                           nBiases.at(0),
                                                                           op.stride(),
                                                                           op.padding(),
                                                                           op.dilation(),
                                                                           op.transposed(),
                                                                           op.output_padding(),
                                                                           op.groups());

                    Value forward = nConv->getResult(1);
                    Value partial = nConv->getResult(0);

                    for(unsigned int i = 1; i < into; i++) {
                        nConv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                    (i == (into-1)) ? TypeRange({returnType}) : TypeRange({returnType, returnType}),
                                                                    forward,
                                                                    partial,
                                                                    nConsts.at(i),
                                                                    nBiases.at(i),
                                                                    op.stride(),
                                                                    op.padding(),
                                                                    op.dilation(),
                                                                    op.transposed(),
                                                                    op.output_padding(),
                                                                    op.groups());
                        partial = nConv->getResult(0);
                        if(i < (into-1)) {
                            forward = nConv->getResult(1);
                        }

                    }


                    // Replace output of old convolution usage by concat value
                    op.getResult().replaceAllUsesWith(nConv->getResult(0));

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

