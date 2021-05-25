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
#include "AirDataflowExplorer.h"

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <string>

#define DEBUG_TYPE "air-dataflow-pass"

#define AInAttr "AIn"
#define AOutAttr "AOut"

using namespace mlir;

// TODO write verifiers
// TODO make sure it works when no weights or no biases but still called on it
// TODO separate hasWeights and hasBias?
// TODO Need a generic WTransform implementation

// General idea: because it is easy to know from the analytical model what do we do with everything:
// That analytical model part is responsible for finding an optimal solution, and then it communicates it here
// Hence in that class we only try to make sure it is easy to generate a graph for any given input

namespace xilinx {
    namespace air {

        struct AirDataflowPass : public PassWrapper<AirDataflowPass, OperationPass<ModuleOp>> {
        private:
            // TODO make the second thing here a map from id based on model params to AbsOpWrapper
            std::map<std::string, std::vector<AbsOpWrapper*>> layerNameToOps;
            std::map<std::string, ModelParams> layerNameToParams;
            std::vector<std::string> layerOrdering;

        public:
            AirDataflowPass() {}

            AbsOpWrapper* opToWrapper(Operation* op) {
                if(auto conv = llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    return new Conv2dReLUOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dReLUOp>(op)) {
                    return new PartialConv2dReLUOpWrapper(conv);
                } else if(auto maxpool = llvm::dyn_cast<mlir::NPCOMP::aten::MaxPool2dWithIndicesOp>(op)) {
                    return new MaxPool2dWithIndicesOpWrapper(maxpool);
                } else if(auto conv = llvm::dyn_cast<Conv2dOp>(op)) {
                    return new Conv2dOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dOp>(op)) {
                    return new PartialConv2dOpWrapper(conv);
                } else if(llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    llvm::outs() << "Unimplemented errror\n";
                    exit(1);
                }
            }

            DataflowExplorer initializeLayerNameToOps(FuncOp graph) {
                // TODO do we have the guarantee to be in network order?
                std::vector<std::pair<std::string, AbsOpWrapper*>> explorerInit;

                graph.walk([&](Operation *op) {
                        if(op->getAttr("name") != nullptr) {
                            auto opName = (op->getAttr("name").dyn_cast<StringAttr>()).getValue();
                            AbsOpWrapper* wrappedOp = opToWrapper(op);
                            if(layerNameToOps.count(opName.str()) == 0) {
                                layerNameToOps[opName.str()] = std::vector<AbsOpWrapper*>({wrappedOp});
                                explorerInit.push_back(std::make_pair(opName.str(), wrappedOp));
                                this->layerNameToParams[opName.str()] = ModelParams(0,0,0,0);
                                this->layerOrdering.push_back(opName.str());
                            } else {
                                llvm::outs() << "Cannot have multiple layer with the same name during initizalization\n";
                                exit(1);
                            }
                        }
                    });

                return DataflowExplorer(explorerInit);
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

            // TODO how to make that generic with respect to Conv Ops? As much as possible?
            LogicalResult PTransform(std::string layerName, unsigned int into) {
                if(into == 1) {
                    return success();
                }

                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    //Conv2dReLUOp op = llvm::dyn_cast<Conv2dReLUOp>(genOp);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;
                    std::vector<Value> nInputs;
                    std::vector<ArrayRef<Value>> nBN;

                    // Split weights
                    Operation* weights;
                    if(genOp->hasWeights()) {
                        weights = genOp->getWeights().getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                            splitConstantInto(constOp, nConsts, builder, PSplit, wSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }

                    // Split biases
                    Operation* biases;
                    if(genOp->hasWeights()) {
                        biases = genOp->getBiases().getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                            splitConstantInto(constOp, nBiases, builder, PSplit, bSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }

                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        std::vector<std::vector<Value>> nBnVect;
                        for(unsigned int i = 0; i < 4; i++) {
                            Operation* bnParam = bnParams[i].getDefiningOp();
                            std::vector<Value> nBnLoc;
                            if(auto constOp = llvm::dyn_cast<ConstantOp>(bnParam)) {
                                splitConstantInto(constOp, nBnLoc, builder, PSplit, bSplitType, into);
                            } else {
                                llvm::outs() << "Cannot convert to ConstOp!\n";
                                return failure();
                            }

                            for(unsigned int j = 0; j < nBnLoc.size(); j++) {
                                if(j == nBnVect.size()) {
                                    nBnVect.push_back(std::vector<Value>({nBnLoc.at(j)}));
                                } else {
                                    nBnVect.at(j).push_back(nBnLoc.at(j));
                                }
                            }
                        }

                        for(auto vect : nBnVect) {
                            nBN.push_back(ArrayRef<Value>({vect.at(0), vect.at(1), vect.at(2), vect.at(3)}));
                        }
                    }


                    // Split Return Type shape
                    // TODO for Maxpool2d with indices, check other return types and check that assumption
                    std::vector<Type> shapes = std::vector<Type>();
                    for(unsigned int i = 0; i < op->getNumResults(); i++) {
                        ShapedType origType = op->getResult(i).getType().dyn_cast<ShapedType>();
                        assert(origType);
                        ShapedType nReturnType = breakShapeInto(origType, C_LOC, into);
                        shapes.push_back(nReturnType);
                    }

                    if(genOp->isDepthWise()) {
                        if(ConcatOp concatOp = genOp->getInput().getDefiningOp<ConcatOp>()) {
                            replaceConcat(builder, concatOp, nInputs, toDelete, C_LOC, into);
                        } else {
                            insertSplit(builder, genOp->getInput(), nInputs, C_LOC, into);
                        }
                    }

                    ArrayRef<Type> nReturnType = ArrayRef<Type>(shapes);

                    // Generate new convs
                    for(unsigned int i = 0; i < into; i++) {
                        Operation* conv;

                        Value input;
                        if(genOp->isDepthWise()) {
                            input = nInputs.at(i);
                        } else {
                            input = genOp->getInput();
                        }

                        auto nW = genOp->hasWeights() ? llvm::Optional<Value>(nConsts.at(i)) : llvm::Optional<Value>();
                        auto nB = genOp->hasWeights() ? llvm::Optional<Value>(nBiases.at(i)) : llvm::Optional<Value>();
                        auto nBn = genOp->hasBN() ? llvm::Optional<ArrayRef<Value>>(nBN.at(i)) : llvm::Optional<ArrayRef<Value>>();
                        conv = genOp->buildOp(builder,
                                              TypeRange({nReturnType}),
                                              input,
                                              nW,
                                              nB,
                                              llvm::Optional<Value>(), false, nBn);

                        assert(conv != nullptr);

                        // set location attribute
                        if(op->getAttr("locP") != nullptr) {
                            auto locP = (op->getAttr("locP").dyn_cast<IntegerAttr>()).getValue();
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, locP + i);
                            conv->setAttr(llvm::StringRef("locP"), attr);
                        } else {
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, i);
                            conv->setAttr(llvm::StringRef("locP"), attr);
                        }

                        nConvs.push_back(conv->getResult(0));
                        nLayerOps.push_back(opToWrapper(conv));
                    }

                    // if split afterwards check size of concat else concat
                    if(op->hasOneUse() && (llvm::dyn_cast<SplitOp>(*(op->getUsers().begin())))) {
                        SplitOp split = llvm::dyn_cast<SplitOp>(*(op->getUsers().begin()));
                        replaceSplit(builder, split, nConvs, toDelete, C_LOC);
                    } else {
                        insertConcat(builder, op->getResult(0), nConvs, C_LOC);
                    }

                    // Delete previous Csts and ConvolutionOp
                    if(genOp->hasWeights()) {
                        toDelete.push_back(weights);
                        toDelete.push_back(biases);
                    }
                }

                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(toDelete);
                deleteOpsFrom(layerOps);


                return success();
            }

            LogicalResult CaTransform(std::string layerName, unsigned int into) {
                if(into == 1) {
                    return success();
                }

                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nInputs;
                    std::vector<ArrayRef<Value>> nBN;

                    // Split weights
                    Operation* weights = genOp->getWeights().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, CaSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                        return failure();
                    }

                    // Split biases
                    Operation* biases = genOp->getBiases().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, CaSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                        return failure();
                    }

                    // Split BN params if any
                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        std::vector<std::vector<Value>> nBnVect;
                        for(unsigned int i = 0; i < 4; i++) {
                            Operation* bnParam = bnParams[i].getDefiningOp();
                            std::vector<Value> nBnLoc;
                            if(auto constOp = llvm::dyn_cast<ConstantOp>(bnParam)) {
                                splitConstantInto(constOp, nBnLoc, builder, CaSplit, bSplitType, into);
                            } else {
                                llvm::outs() << "Cannot convert to ConstOp!\n";
                                return failure();
                            }

                            for(unsigned int j = 0; j < nBnLoc.size(); j++) {
                                if(j == nBnVect.size()) {
                                    nBnVect.push_back(std::vector<Value>({nBnLoc.at(j)}));
                                } else {
                                    nBnVect.at(j).push_back(nBnLoc.at(j));
                                }
                            }
                        }

                        for(auto vect : nBnVect) {
                            nBN.push_back(ArrayRef<Value>({vect.at(0), vect.at(1), vect.at(2), vect.at(3)}));
                        }
                    }

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
                    auto bn = genOp->hasBN() ? llvm::Optional<ArrayRef<Value>>(nBN.at(0)) : llvm::Optional<ArrayRef<Value>>();
                    Operation* conv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                     nInputs.at(0), llvm::Optional<Value>(nConsts.at(0)),
                                                     llvm::Optional<Value>(nBiases.at(0)), llvm::Optional<Value>(), true, bn);

                    // set location attribute
                    if(op->getAttr("locCa") != nullptr) {
                        auto locCa = (op->getAttr("locCa").dyn_cast<IntegerAttr>()).getValue();
                        auto ty = IntegerType::get(builder.getContext(), 32);
                        auto attr = IntegerAttr::get(ty, locCa);
                        conv->setAttr(llvm::StringRef("locCa"), attr);
                    } else {
                        auto ty = IntegerType::get(builder.getContext(), 32);
                        auto attr = IntegerAttr::get(ty, 0);
                        conv->setAttr(llvm::StringRef("locCa"), attr);
                    }

                    nLayerOps.push_back(opToWrapper(conv));

                    for(unsigned int i = 1; i < into; i++) {
                        auto bn = genOp->hasBN() ? llvm::Optional<ArrayRef<Value>>(nBN.at(i)) : llvm::Optional<ArrayRef<Value>>();
                        Operation* nConv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                          nInputs.at(i), llvm::Optional<Value>(nConsts.at(i)),
                                                          llvm::Optional<Value>(nBiases.at(i)),
                                                          llvm::Optional<Value>(conv->getResult(0)), true, bn);

                        // set location attribute
                        if(op->getAttr("locCa") != nullptr) {
                            auto locCa = (op->getAttr("locCa").dyn_cast<IntegerAttr>()).getValue();
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, locCa + i);
                            nConv->setAttr(llvm::StringRef("locCa"), attr);
                        } else {
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, i);
                            nConv->setAttr(llvm::StringRef("locCa"), attr);
                        }

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
                if(into == 1) {
                    return success();
                }
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;
                    std::vector<ArrayRef<Value>> nBN;

                    // Split weights
                    Operation* weights = genOp->getWeights().getDefiningOp();//->getName();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                        splitConstantInto(constOp, nConsts, builder, LSplit, wSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                        return failure();
                    }

                    // Split biases
                    Operation* biases = genOp->getBiases().getDefiningOp();
                    if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                        splitConstantInto(constOp, nBiases, builder, LSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                        return failure();
                    }

                    // Split BN params if present
                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        std::vector<std::vector<Value>> nBnVect;
                        for(unsigned int i = 0; i < 4; i++) {
                            Operation* bnParam = bnParams[i].getDefiningOp();
                            std::vector<Value> nBnLoc;
                            if(auto constOp = llvm::dyn_cast<ConstantOp>(bnParam)) {
                                splitConstantInto(constOp, nBnLoc, builder, LSplit, bSplitType, into);
                            } else {
                                llvm::outs() << "Cannot convert to ConstOp!\n";
                                return failure();
                            }

                            for(unsigned int j = 0; j < nBnLoc.size(); j++) {
                                if(j == nBnVect.size()) {
                                    nBnVect.push_back(std::vector<Value>({nBnLoc.at(j)}));
                                } else {
                                    nBnVect.at(j).push_back(nBnLoc.at(j));
                                }
                            }
                        }

                        for(auto vect : nBnVect) {
                            nBN.push_back(ArrayRef<Value>({vect.at(0), vect.at(1), vect.at(2), vect.at(3)}));
                        }
                    }

                    // Same return type here
                    ShapedType retType = op->getResult(0).getType().dyn_cast<ShapedType>();

                    // Generate new convs
                    auto bn = genOp->hasBN() ? llvm::Optional<ArrayRef<Value>>(nBN.at(0)) : llvm::Optional<ArrayRef<Value>>();
                    Operation* nConv = genOp->buildOp(builder, TypeRange({retType, retType}),
                                                      genOp->getInput(), llvm::Optional<Value>(nConsts.at(0)),
                                                      llvm::Optional<Value>(nBiases.at(0)), llvm::Optional<Value>(), true,
                                                      bn);

                    // set location attribute
                    if(op->getAttr("locL") != nullptr) {
                        auto locL = (op->getAttr("locL").dyn_cast<IntegerAttr>()).getValue();
                        auto ty = IntegerType::get(builder.getContext(), 32);
                        auto attr = IntegerAttr::get(ty, locL);
                        nConv->setAttr(llvm::StringRef("locL"), attr);
                    } else {
                        auto ty = IntegerType::get(builder.getContext(), 32);
                        auto attr = IntegerAttr::get(ty, 0);
                        nConv->setAttr(llvm::StringRef("locL"), attr);
                    }

                    Value forward = nConv->getResult(1);
                    Value partial = nConv->getResult(0);
                    nLayerOps.push_back(opToWrapper(nConv));

                    for(unsigned int i = 1; i < into; i++) {
                        auto bn = genOp->hasBN() ? llvm::Optional<ArrayRef<Value>>(nBN.at(i)) : llvm::Optional<ArrayRef<Value>>();
                        nConv = genOp->buildOp(builder,
                                               (i == (into-1)) ? TypeRange({retType}) : TypeRange({retType, retType}),
                                               forward, llvm::Optional<Value>(nConsts.at(i)),
                                               llvm::Optional<Value>(nBiases.at(i)),
                                               llvm::Optional<Value>(partial), false,
                                               bn);

                        // set location attribute
                        if(op->getAttr("locL") != nullptr) {
                            auto locL = (op->getAttr("locL").dyn_cast<IntegerAttr>()).getValue();
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, locL + i);
                            nConv->setAttr(llvm::StringRef("locL"), attr);
                        } else {
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, i);
                            nConv->setAttr(llvm::StringRef("locL"), attr);
                        }

                        partial = nConv->getResult(0);
                        if(i < (into-1)) {
                            forward = nConv->getResult(1);
                        }
                        nLayerOps.push_back(opToWrapper(nConv));

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

            unsigned int getAttrOrDefault(Operation* op, std::string attrName, unsigned int defVal) {
                if(op->getAttr(attrName) != nullptr) {
                    return op->getAttr(attrName).dyn_cast<IntegerAttr>().getValue().getZExtValue();
                } else {
                    return defVal;
                }
            }

            // TODO take into account depthwise layers
            // TODO work at the tile grannularity
            // TODO probably do not care of the line stuff even: show this better
            std::vector<std::string> workOn(Operation* op, DataflowExplorer &expl) {
                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                unsigned int locP = getAttrOrDefault(op, "locP", 0);

                std::string layerName = op->getAttr("name").dyn_cast<StringAttr>().getValue().str();
                uint64_t tilesPerCore = expl.getTilesPerCore(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);

                std::vector<std::string> locLines;
                for(uint64_t i = 0; i < tilesPerCore; i++) {
                    locLines.push_back("tile" + std::to_string(locL + locW + i) + "C" + std::to_string(locCa));
                }

                return locLines;
            }

            std::vector<std::string> wantLoc(Operation* op, DataflowExplorer &expl) {
                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                unsigned int locP = getAttrOrDefault(op, "locP", 0);

                std::string layerName = op->getAttr("name").dyn_cast<StringAttr>().getValue().str();

                uint64_t tilesPerCore = expl.getTilesPerCore(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);
                unsigned int W = this->layerNameToParams[layerName].W;

                uint64_t startLine = locL + locW;
                uint64_t endLine = locL + locW + tilesPerCore - 1;

                std::vector<std::string> wantLines;
                for(uint64_t i = 0; i < tilesPerCore; i++) {
                    uint64_t wantLine = locL + locW + W + i;
                    if((wantLine < startLine) || (wantLine > endLine)) {
                        wantLines.push_back("tile" + std::to_string(locL + locW + W) + "C" + std::to_string(locCa));
                    }
                }

                return wantLines;
            }

            std::vector<std::string> wantPrev(Operation* op, DataflowExplorer &expl) {
                std::string layerName = op->getAttr("name").dyn_cast<StringAttr>().getValue().str();
                unsigned int W = this->layerNameToParams[layerName].W;
                unsigned int L = this->layerNameToParams[layerName].L;

                uint64_t tilesPerCore = expl.getTilesPerCore(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);
                unsigned int highestLoc = (W-1) + (L-1);

                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                unsigned int locP = getAttrOrDefault(op, "locP", 0);

                uint64_t startLine = locL + locW;
                uint64_t endLine = locL + locW + tilesPerCore - 1;

                std::vector<std::string> wantLines;
                for(uint64_t i = 0; i < tilesPerCore; i++) {
                    uint64_t wantLine = locL + locW + W + i;
                    if(((wantLine < startLine) || (wantLine > endLine)) && (wantLine - highestLoc >= 0)) {
                        wantLines.push_back("tile" + std::to_string(wantLine - highestLoc) + "C" + std::to_string(locCa));
                    }
                }

                return wantLines;
                //unsigned int wantLine = locL + locW + W - highestLoc;
                //return "tile" + std::to_string(wantLine) + "C" + std::to_string(locCa);
            }

            // TODO Double check depth-wise handling
            std::map<std::string, Value> findProducedTiles(std::string layerName) {
                std::map<std::string, Value> producedLineToOp;
                ModelParams params = this->layerNameToParams[layerName];
                for(AbsOpWrapper* prevAbsOp : this->layerNameToOps[layerName]) {
                    Operation* op = prevAbsOp->getUnderlyingOperation();
                    unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                    unsigned int locL = getAttrOrDefault(op, "locL", 0);
                    unsigned int locW = getAttrOrDefault(op, "locW", 0);
                    unsigned int locP = getAttrOrDefault(op, "locP", 0);

                    if(locCa == (params.Ca-1) && locL == (params.L-1)) { // is a producer
                        if(op->getResult(0).hasOneUse() && llvm::dyn_cast<ConcatOp>(*(op->getResult(0).getUsers().begin()))) {
                            ConcatOp concat = llvm::dyn_cast<ConcatOp>(*(op->getResult(0).getUsers().begin()));

                            // check if is a concat from P and not from W
                            unsigned int concatW = (unsigned int)-1;
                            unsigned int concatP = (unsigned int)-1;
                            bool isPConcat = true;
                            for(auto o : concat.getOperands()) {
                                Operation* concatArg = o.getDefiningOp();
                                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                                unsigned int locP = getAttrOrDefault(op, "locP", 0);

                                if(locP < concatP) {
                                    concatP = locP;
                                }

                                if(locW == (unsigned int)-1) {
                                    concatW = locW;
                                } else if(locW != concatW) {
                                    isPConcat = false;
                                    break;
                                }
                            }

                            unsigned int concatSize = concat.getNumOperands();
                            if(isPConcat) {
                                std::string hashString = "tile" + std::to_string(concatW) + "P" + std::to_string(concatP / concatSize);
                                producedLineToOp[hashString] = concat.getResult();
                            }
                        } else if(op->getResult(0).hasOneUse() && llvm::dyn_cast<SplitOp>(*(op->getResult(0).getUsers().begin()))) {
                            SplitOp split = llvm::dyn_cast<SplitOp>(*(op->getResult(0).getUsers().begin()));

                            // Check if this split is from a P not from W
                            unsigned int splitW = (unsigned int)-1;
                            bool isPSplit = true;
                            for(auto u : split->getUsers()) { // TODO double check this
                                unsigned int locW = getAttrOrDefault(u, "locW", 0);

                                if(locW == (unsigned int)-1) {
                                    splitW = locW;
                                } else if(locW != splitW) {
                                    isPSplit = false;
                                    break;
                                }
                            }

                            unsigned int splitSize = split.getNumResults();
                            unsigned int locP = getAttrOrDefault(split.getOperation(), "locP", 0);
                            if(isPSplit) {
                                for(unsigned int i = 0; i < splitSize; i++) {
                                    unsigned int pLoc = locP * splitSize + i;
                                    std::string hashString = "tile" + std::to_string(splitW) + "P" + std::to_string(pLoc);
                                    producedLineToOp[hashString] = split.getResult(i);
                                }
                            }
                        } else {
                            std::string hashString = "tile" + std::to_string(locW) + "P" + std::to_string(locP);
                            llvm::outs() << "hs: " << hashString << "\n";
                            producedLineToOp[hashString] = prevAbsOp->getUnderlyingOperation()->getResult(0);
                        }
                    }
                }

                return producedLineToOp;
            }

            // TODO for now select arbitrary line from any core that has it, might change that
            std::map<std::string, Value> findLocalTiles(std::string layerName, DataflowExplorer &expl) {
                std::map<std::string, Value> localLines;
                ModelParams params = this->layerNameToParams[layerName];
                for(AbsOpWrapper* absOp : this->layerNameToOps[layerName]) {
                    std::vector<std::string> linesLoc = this->workOn(absOp->getUnderlyingOperation(), expl);
                    for(std::string s : linesLoc) {
                        localLines[s] = absOp->getUnderlyingOperation()->getResult(1);
                    }
                }

                return localLines;
            }

            // TODO also duplicate concat and splits
            void wDuplicate(std::string layerName, unsigned int into) {
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];

                for(int64_t i = into-1; i >= 0; i--) {
                    OpBuilder builder(layerNameToOps[layerName].at(0)->getUnderlyingOperation());
                    std::map<std::string, AbsOpWrapper*> paramsToLayer;
                    std::map<std::string, std::vector<Value>> concatLocToArg;
                    std::map<std::string, Value> concatLocToRes;
                    for(AbsOpWrapper* absOp : layerNameToOps[layerName]) {
                        if(i == 0) {
                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, 0);
                            absOp->getUnderlyingOperation()->setAttr(llvm::StringRef("locW"), attr);
                        } else {
                            Operation* op = absOp->wCopy(builder, i);

                            auto ty = IntegerType::get(builder.getContext(), 32);
                            auto attr = IntegerAttr::get(ty, i);
                            op->setAttr(llvm::StringRef("locW"), attr);

                            AbsOpWrapper* locAbsOp = opToWrapper(op);
                            layerOps.push_back(locAbsOp);

                            Value opRes = absOp->getUnderlyingOperation()->getResult(0);
                            if(opRes.hasOneUse() && llvm::dyn_cast<SplitOp>(*(opRes.getUsers().begin()))) {
                                // Do not link back splits as will be done later?
                                SplitOp split = llvm::dyn_cast<SplitOp>(*(opRes.getUsers().begin()));
                                std::vector<Value> tmp;
                                insertSplit(builder, opRes, tmp, C_LOC, split.getNumResults());
                            }

                            if(opRes.hasOneUse() && llvm::dyn_cast<ConcatOp>(*opRes.getUsers().begin())) {
                                ConcatOp concat = llvm::dyn_cast<ConcatOp>(*opRes.getUsers().begin());
                                unsigned int concatP = (unsigned int)-1;
                                unsigned int concatW = (unsigned int)-1;

                                for(auto v : concat.getOperands()) {
                                    unsigned int locP = getAttrOrDefault(v.getDefiningOp(), "locP", 0);
                                    if(concatP > locP) {
                                        concatP = locP;
                                    }

                                    concatW = getAttrOrDefault(v.getDefiningOp(), "locW", 0);
                                }

                                unsigned int locP = getAttrOrDefault(op, "locP", 0);
                                unsigned int vectorLoc = locP - concatP;

                                std::string hashString = "tile" + std::to_string(concatW) + "P" + std::to_string(concatP);
                                unsigned int concatLocToArgSize = concatLocToArg[hashString].size();

                                if(concatLocToArgSize == vectorLoc) {
                                    concatLocToArg[hashString].push_back(op->getResult(0));
                                } else if(concatLocToArgSize < vectorLoc) {
                                    for(unsigned int i = concatLocToArgSize; i < vectorLoc; i++) {
                                        concatLocToArg[hashString].push_back(Value());
                                    }

                                    concatLocToArg[hashString].push_back(op->getResult(0));
                                } else {
                                    concatLocToArg[hashString][vectorLoc] = op->getResult(0);
                                }

                                concatLocToRes[hashString] = concat.getResult();
                            }

                            unsigned int locCa = getAttrOrDefault(absOp->getUnderlyingOperation(), "locCa", 0);
                            unsigned int locL = getAttrOrDefault(absOp->getUnderlyingOperation(), "locL", 0);
                            unsigned int locP = getAttrOrDefault(absOp->getUnderlyingOperation(), "locP", 0);
                            std::string hashString = "P" + std::to_string(locP) +
                                "Ca" + std::to_string(locCa) +
                                "L" + std::to_string(locL);

                            paramsToLayer[hashString] = locAbsOp;
                        }
                    }

                    // Re-wire duplicated one with inputs from same W group
                    std::map<std::string, AbsOpWrapper*>::iterator it;
                    for(it = paramsToLayer.begin(); it != paramsToLayer.end(); it++) {
                        AbsOpWrapper* absOp = it->second;
                        Operation* op = absOp->getUnderlyingOperation();

                        unsigned int locCa = getAttrOrDefault(absOp->getUnderlyingOperation(), "locCa", 0);
                        unsigned int locL = getAttrOrDefault(absOp->getUnderlyingOperation(), "locL", 0);
                        unsigned int locP = getAttrOrDefault(absOp->getUnderlyingOperation(), "locP", 0);

                        unsigned int Ca = this->layerNameToParams[layerName].Ca;
                        unsigned int P = this->layerNameToParams[layerName].P;
                        unsigned int L = this->layerNameToParams[layerName].L;

                        if(locL != 0) {
                            std::string hashString = "P" + std::to_string(locP)
                                + "Ca" + std::to_string(locCa)
                                + "L" + std::to_string(locL - 1);
                            AbsOpWrapper* prevAbsOp = paramsToLayer[hashString];

                            op->replaceUsesOfWith(absOp->getInput(), prevAbsOp->getUnderlyingOperation()->getResult(1));
                            op->replaceUsesOfWith(absOp->getPartialInput(), prevAbsOp->getUnderlyingOperation()->getResult(0));
                        } else if(locCa != 0  && locL == 0) {
                            std::string hashString = "P" + std::to_string(locP)
                                + "Ca" + std::to_string(locCa-1)
                                + "L" + std::to_string(L-1);
                            AbsOpWrapper* prevAbsOp = paramsToLayer[hashString];

                            //op->replaceUsesOfWith(absOp->getInput(), prevAbsOp->getUnderlyingOperation()->getResult(1));
                            op->replaceUsesOfWith(absOp->getPartialInput(), prevAbsOp->getUnderlyingOperation()->getResult(0));
                        }
                    }

                    // Instantiate the duplicated concats
                    std::map<std::string, std::vector<Value>>:: iterator concatIt;
                    for(concatIt = concatLocToArg.begin(); concatIt != concatLocToArg.end(); concatIt++) {
                        insertConcat(builder, concatLocToRes[it->first], concatIt->second, C_LOC);
                    }
                }

                // Assign new layer
                layerNameToOps[layerName] = layerOps;
            }

            void reWire(std::string layerName, DataflowExplorer &expl) {
                OpBuilder builder(layerNameToOps[layerName].at(0)->getUnderlyingOperation());

                // construct line location
                std::vector<std::string>::iterator layerLoc;
                layerLoc = std::find(this->layerOrdering.begin(), this->layerOrdering.end(), layerName);
                bool firstLayer = layerLoc == this->layerOrdering.begin();

                std::map<std::string, Value> producedTiles;
                if(!firstLayer) {
                    producedTiles = this->findProducedTiles(*(layerLoc-1));

                    // Makes sure producedTile Shape matches with the one of the current layer
                    ModelParams paramsCurr = this->layerNameToParams[layerName];
                    ModelParams paramsPrev = this->layerNameToParams[expl.layerIdToName[expl.layerNameToID[layerName]-1]];

                    if(paramsCurr.W > paramsPrev.W) { // Duplicate so that matches next
                        unsigned int ratio = ceil((float)paramsCurr.W / paramsPrev.W);

                        for(unsigned int p = 0; p < paramsPrev.P; p++) {
                            for(unsigned int i = 0; i < ratio; i++) {
                                for(unsigned int w = 0; w < paramsPrev.W; w++) {
                                    std::string hashString = "tile" + std::to_string(w) + "P" + std::to_string(p);
                                    std::string dupHashString = "tile" + std::to_string(w + i * ratio) + "P" + std::to_string(p);

                                    producedTiles[dupHashString] = producedTiles[hashString];
                                }
                            }
                        }
                    } else if(paramsCurr.W < paramsPrev.W) { // Insert concat on parallel tiles produced
                        unsigned int ratio = ceil((float)paramsPrev.W / paramsCurr.W);

                        std::vector<Value> concats;
                        std::vector<Value> concatsArgs;

                        for(unsigned int p = 0; p < paramsPrev.P; p++) {
                            for(unsigned int i = 0; i < paramsPrev.W; i++) {
                                if((i != 0) && ((i % ratio) == 0)) {
                                    concats.push_back(insertConcat(builder, concatsArgs.at(0), concatsArgs, N_LOC)->getResult(0));
                                    concatsArgs.clear();
                                } else {
                                    std::string hashString = "tile" + std::to_string(i) + "P" + std::to_string(p);
                                    concatsArgs.push_back(producedTiles[hashString]);
                                }
                            }

                            if(concatsArgs.size() != 0) {
                                concats.push_back(insertConcat(builder, concatsArgs.at(0), concatsArgs, N_LOC)->getResult(0));
                                concatsArgs.clear();
                            }

                            for(unsigned int i = 0; i < concats.size(); i++) {
                                std::string hashString = "tile" + std::to_string(i) + "P" + std::to_string(p);
                                producedTiles[hashString] = concats.at(i);
                            }

                            for(unsigned int i = concats.size(); i < paramsPrev.W; i++) {
                                std::string hashString = "tile" + std::to_string(i) + "P" + std::to_string(p);
                                producedTiles.erase(hashString);
                            }
                        }
                    }

                    // TODO remove potential Wconcat stuff that was there before
                    // TODO or leave it to a potential clean pass
                }

                std::map<std::string, Value> locTiles = this->findLocalTiles(layerName, expl);

                // really re-wire from reconstructed info
                // Now easy because guarantee to find exactly what we need
                for(AbsOpWrapper* absOp : this->layerNameToOps[layerName]) {
                    Operation* op = absOp->getUnderlyingOperation();
                    if(firstLayer) {
                        // TODO link to input of network
                    } else {
                        std::vector<std::string> wantLoc = this->wantLoc(op, expl);
                        for(std::string s : wantLoc) {
                            if(locTiles.find(s) != locTiles.end()) {
                                op->replaceUsesOfWith(absOp->getInput(), locTiles[s]);
                            }
                        }

                        std::vector<std::string> wantPrev = this->wantPrev(op, expl);
                        for(std::string s : wantPrev) {
                            op->replaceUsesOfWith(absOp->getInput(), producedTiles[s]);
                        }
                    }
                }

            }

            // TODO might generate chains of concat, or concat and then split on a different dim
            // TODO either need a simplify pass of handle things better
            LogicalResult WTransform(std::string layerName, unsigned int into, DataflowExplorer &expl) {
                if(into == 1) {
                    return success();
                }

                // duplicate graph into times
                wDuplicate(layerName, into);

                // Re-wire
                reWire(layerName, expl);
                if(expl.layerNameToID[layerName] != (expl.layerNameToID.size()-1)) {
                    reWire(expl.layerIdToName[expl.layerNameToID[layerName]+1], expl);
                }

                return success();
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<FuncOp>("graph");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Cant find graph func\n");
                    signalPassFailure();
                    return;
                }

                DataflowExplorer dataflowExplorer = initializeLayerNameToOps(graph);
                //initializeLayerNameToParams(graph);

                // Explore topology space
                dataflowExplorer.enumerate();
                //dataflowExplorer.printValidTopologies();
                dataflowExplorer.dumpValidTopologies();
                dataflowExplorer.dumpParetoFrontiers();
                dataflowExplorer.dumpPathsFrom(dataflowExplorer.paretoThroughput, "./output/throughput");
                dataflowExplorer.dumpPathsFrom(dataflowExplorer.paretoLatency, "./output/latency");

                this->layerNameToParams = dataflowExplorer.getMaxThroughput();

                // Expand P, Ca, L for all layers
                /*std::map<std::string, ModelParams>::iterator it;
                  for(it = layerNameToParams.begin(); it != layerNameToParams.end(); it++) {
                  unsigned int P = it->second.P;
                  unsigned int Ca = it->second.Ca;
                  unsigned int L = it->second.L;

                  if(!PTransform(it->first, P).succeeded()) {
                  llvm::outs() << "Failed to apply PTransform\n";
                  exit(1);
                  }

                  if(!CaTransform(it->first, Ca).succeeded()) {
                  llvm::outs() << "Failed to apply CaTransform\n";
                  exit(1);
                  }

                  if(!LTransform(it->first, L).succeeded()) {
                  llvm::outs() << "Failed to apply LTransform\n";
                  exit(1);
                  }
                  }

                  this->layerNameToParams["conv2d_relu0"] = ModelParams(1,1,1,1);
                  this->layerNameToParams["conv2d_relu1"] = ModelParams(1,1,3,3);
                  this->layerNameToParams["conv2d_relu2"] = ModelParams(1,1,3,1);

                  this->layerNameToParams["max_pool2d_with_indices0"] = ModelParams(1,1,1,1);
                  this->layerNameToParams["max_pool2d_with_indices1"] = ModelParams(1,1,1,1);

                  //PTransform("conv2d_relu1", 4);
                  //PTransform("max_pool2d_with_indices1", 4);
                  //CaTransform("conv2d_relu2", 4);
                  LTransform("conv2d_relu1", 3);
                  LTransform("conv2d_relu2", 3);

                  // annotate lines
                  annotateLines();

                  // W expand
                  WTransform("conv2d_relu1", 3);*/

                // Verify graph

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

