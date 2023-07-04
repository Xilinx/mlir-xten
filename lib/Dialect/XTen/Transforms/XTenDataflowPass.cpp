//===- XTenDataflowPass.cpp -------------------------------------*- C++ -*-===//
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
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "xten/Dialect/XTen/XTenDataflow.h"
#include "xten/Dialect/XTen/XTenDataflowUtils.h"
#include "xten/Dialect/XTen/XTenDataflowExplorer.h"

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <string>

#define DEBUG_TYPE "xten-dataflow-pass"

#define AInAttr "AIn"
#define AOutAttr "AOut"

using namespace mlir;

// TODO write verifiers
// TODO properly take stride into account when building the graph, especially for W

// General idea: because it is easy to know from the analytical model what do we do with everything:
// That analytical model part is responsible for finding an optimal solution, and then it communicates it here
// Hence in that class we only try to make sure it is easy to generate a graph for any given input

namespace xilinx {
    namespace xten {

        struct XTenDataflowPass : public XTenDataflowBase<XTenDataflowPass> {
        private:
            // TODO make the second thing here a map from id based on model params to AbsOpWrapper
            std::map<std::string, std::vector<AbsOpWrapper*>> layerNameToOps;
            std::map<std::string, ModelParams> layerNameToParams;
            std::vector<std::string> layerOrdering; // TODO remove this field

        public:
            XTenDataflowPass() {}

            AbsOpWrapper* opToWrapper(Operation* op) {
                if(auto conv = llvm::dyn_cast<Conv2dReLUOp>(op)) {
                    return new Conv2dReLUOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dReLUOp>(op)) {
                    return new PartialConv2dReLUOpWrapper(conv);
                } else if(auto maxpool = llvm::dyn_cast<torch::Torch::AtenMaxPool2dOp>(op)) {
                    return new MaxPool2dOpWrapper(maxpool);
                } else if(auto conv = llvm::dyn_cast<Conv2dOp>(op)) {
                    return new Conv2dOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dOp>(op)) {
                    return new PartialConv2dOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<Conv2dBatchNormReLUOp>(op)) {
                    return new Conv2dBatchNormReLUOpWrapper(conv);
                } else if(auto conv = llvm::dyn_cast<PartialConv2dBatchNormReLUOp>(op)) {
                    return new PartialConv2dBatchNormReLUOpWrapper(conv);
                } else {
                    llvm::outs() << "Unsupported operation was used!\n";
                    exit(1);
                }
            }

            DataflowExplorer initializeLayerNameToOps(func::FuncOp graph) {
                // TODO do we have the guarantee to be in network order?
                std::vector<std::pair<std::string, AbsOpWrapper*>> explorerInit;

                graph.walk([&](Operation *op) {
                        if(op->getAttr("layer_name") != nullptr) {
                            auto opName = (op->getAttr("layer_name").dyn_cast<StringAttr>()).getValue();
                            AbsOpWrapper* wrappedOp = opToWrapper(op);
                            if(layerNameToOps.count(opName.str()) == 0) {
                                layerNameToOps[opName.str()] = std::vector<AbsOpWrapper*>({wrappedOp});
                                explorerInit.push_back(std::make_pair(opName.str(), wrappedOp));
                                this->layerNameToParams[opName.str()] = ModelParams(0,0,0,0,false);
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
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(weights)) {
                            splitConstantInto(constOp, nConsts, builder, PSplit, wSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }

                    // Split biases
                    Operation* biases;
                    if(genOp->hasBias()) {
                        biases = genOp->getBiases()->getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(biases)) {
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
                            if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(bnParam)) {
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
                        mlir::torch::Torch::BaseTensorType origType = op->getResult(i).getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
                        assert(origType);
                        mlir::torch::Torch::BaseTensorType nReturnType = breakShapeInto(origType, C_LOC, into);
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

                        auto nW = genOp->hasWeights() ? std::optional<Value>(nConsts.at(i)) : std::optional<Value>();
                        auto nB = genOp->hasBias() ? std::optional<Value>(nBiases.at(i)) : std::optional<Value>();
                        auto nBn = genOp->hasBN() ? std::optional<ArrayRef<Value>>(nBN.at(i)) : std::optional<ArrayRef<Value>>();
                        conv = genOp->buildOp(builder,
                                              TypeRange({nReturnType}),
                                              input,
                                              nW,
                                              nB,
                                              std::optional<Value>(), false, nBn);

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
                        insertConcat(builder, op->getResult(0), nConvs, C_LOC, true);
                    }

                    // Delete previous Csts and ConvolutionOp
                    if(genOp->hasWeights()) {
                        toDelete.push_back(weights);
                    }

                    if(genOp->hasBias()) {
                        toDelete.push_back(biases);
                    }

                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        for(unsigned int i = 0; i < 4; i++) {
                            toDelete.push_back(bnParams[i].getDefiningOp());
                        }
                    }
                }

                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(layerOps);
                deleteOpsFrom(toDelete);

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
                    Operation* weights;
                    if(genOp->hasWeights()) {
                         weights = genOp->getWeights().getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(weights)) {
                            splitConstantInto(constOp, nConsts, builder, CaSplit, wSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }


                    // Split biases
                    Operation* biases;
                    if(genOp->hasBias()) {
                        biases = genOp->getBiases()->getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(biases)) {
                            splitConstantInto(constOp, nBiases, builder, CaSplit, bSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }


                    // Split BN params if any
                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        std::vector<std::vector<Value>> nBnVect;
                        for(unsigned int i = 0; i < 4; i++) {
                            Operation* bnParam = bnParams[i].getDefiningOp();
                            std::vector<Value> nBnLoc;
                            if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(bnParam)) {
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
                    if(auto constOp = genOp->getInput().getDefiningOp<mlir::arith::ConstantOp>()) {
                        splitConstantInto(constOp, nInputs, builder, CaSplit, aSplitType, into);
                    } else {
                        if(ConcatOp concatOp = genOp->getInput().getDefiningOp<ConcatOp>()) {
                            unsigned int locP = getAttrOrDefault(genOp->getUnderlyingOperation(), "locP", 0);
                            std::vector<Operation*> toDel = (locP == 0) ? toDelete : std::vector<Operation*>(); // TODO clean this
                            replaceConcat(builder, concatOp, nInputs, toDel, C_LOC, into);
                        } else {
                            insertSplit(builder, genOp->getInput(), nInputs, C_LOC, into);
                        }
                    }

                    // Generate convolutions
                    auto w = genOp->hasWeights() ? std::optional<Value>(nConsts.at(0)) : std::optional<Value>();
                    auto bias = genOp->hasBias() ? std::optional<Value>(nBiases.at(0)) : std::optional<Value>();
                    auto bn = genOp->hasBN() ? std::optional<ArrayRef<Value>>(nBN.at(0)) : std::optional<ArrayRef<Value>>();
                    auto chainIn = std::optional<Value>(genOp->getPartialInput());
                    Operation* conv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                     nInputs.at(0), w, bias, chainIn, true, bn);

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
                        auto w = genOp->hasWeights() ? std::optional<Value>(nConsts.at(i)) : std::optional<Value>();
                        auto bias = genOp->hasBias() ? std::optional<Value>(nBiases.at(i)) : std::optional<Value>();
                        auto bn = genOp->hasBN() ? std::optional<ArrayRef<Value>>(nBN.at(i)) : std::optional<ArrayRef<Value>>();
                        Operation* nConv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                          nInputs.at(i), w, bias, std::optional<Value>(conv->getResult(0)), true, bn);

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
                    if(genOp->hasWeights()) {
                        toDelete.push_back(weights);
                    }

                    if(genOp->hasBias()) {
                        toDelete.push_back(biases);
                    }

                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        for(unsigned int i = 0; i < 4; i++) {
                            toDelete.push_back(bnParams[i].getDefiningOp());
                        }
                    }
                }

                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(layerOps);
                deleteOpsFrom(toDelete);

                return success();
            }

            LogicalResult LTransform(std::string layerName, unsigned int into) {
                if(into == 1) {
                    return success();
                }
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];
                std::vector<Operation*> toDelete;
                std::vector<AbsOpWrapper*> nLayerOps;

                llvm::outs() << "LayerOps size: " << layerOps.size();

                for(AbsOpWrapper* genOp : layerOps) {
                    Operation* op = genOp->getUnderlyingOperation();
                    OpBuilder builder(op);

                    std::vector<Value> nConsts;
                    std::vector<Value> nBiases;
                    std::vector<Value> nConvs;
                    std::vector<ArrayRef<Value>> nBN;

                    // Split weights
                    Operation* weights;
                    if(genOp->hasWeights()) {
                        weights = genOp->getWeights().getDefiningOp();//->getName();
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(weights)) {
                            splitConstantInto(constOp, nConsts, builder, LSplit, wSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }


                    // Split biases
                    Operation* biases;
                    if(genOp->hasBias()) {
                        biases = genOp->getBiases()->getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(biases)) {
                            splitConstantInto(constOp, nBiases, builder, LSplit, bSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
                            return failure();
                        }
                    }

                    // Split BN params if present
                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        std::vector<std::vector<Value>> nBnVect;
                        for(unsigned int i = 0; i < 4; i++) {
                            Operation* bnParam = bnParams[i].getDefiningOp();
                            std::vector<Value> nBnLoc;
                            if(auto constOp = llvm::dyn_cast<mlir::arith::ConstantOp>(bnParam)) {
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
                    mlir::torch::Torch::BaseTensorType retTypePartial = op->getResult(0).getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
                    mlir::torch::Torch::BaseTensorType retTypeForward = genOp->getInput().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();

                    // Generate new convs
                    auto w = genOp->hasWeights() ? std::optional<Value>(nConsts.at(0)) : std::optional<Value>();
                    auto bias = genOp->hasBias() ? std::optional<Value>(nBiases.at(0)) : std::optional<Value>();
                    auto bn = genOp->hasBN() ? std::optional<ArrayRef<Value>>(nBN.at(0)) : std::optional<ArrayRef<Value>>();
                    auto chainIn = std::optional<Value>(genOp->getPartialInput());
                    Operation* nConv = genOp->buildOp(builder, TypeRange({retTypePartial, retTypeForward}),
                                                      genOp->getInput(), w, bias, chainIn, true, bn);

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
                        auto w = genOp->hasWeights() ? std::optional<Value>(nConsts.at(i)) : std::optional<Value>();
                        auto bias = genOp->hasBias() ? std::optional<Value>(nBiases.at(i)) : std::optional<Value>();
                        auto bn = genOp->hasBN() ? std::optional<ArrayRef<Value>>(nBN.at(i)) : std::optional<ArrayRef<Value>>();
                        // Same return type here
                        nConv = genOp->buildOp(builder,
                                               (i == (into-1)) ? TypeRange({retTypePartial}) : TypeRange({retTypePartial, retTypeForward}),
                                               forward, w, bias, std::optional<Value>(partial), false, bn);

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

                    if(genOp->hasWeights()) {
                        toDelete.push_back(weights);
                    }

                    if(genOp->hasBias()) {
                        toDelete.push_back(biases);
                    }

                    if(genOp->hasBN()) {
                        ArrayRef<Value> bnParams = genOp->getBN();
                        for(unsigned int i = 0; i < 4; i++) {
                            toDelete.push_back(bnParams[i].getDefiningOp());
                        }
                    }
                }

                // Delete previous Csts and ConvolutionOp
                layerNameToOps[layerName] = nLayerOps;

                // cleanup
                deleteOpsFrom(layerOps);
                deleteOpsFrom(toDelete);

                return success();
            }

            // TODO take into account depthwise layers
            // TODO work at the tile grannularity
            // TODO Support correct line stuff
            std::vector<std::string> workOn(AbsOpWrapper* absOp, DataflowExplorer &expl) {
                Operation* op = absOp->getUnderlyingOperation();

                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                //unsigned int locP = getAttrOrDefault(op, "locP", 0);

                std::string layerName = op->getAttr("layer_name").dyn_cast<StringAttr>().getValue().str();

                uint64_t F0 = absOp->getF0();
                // TODO fix for F1

                uint64_t linesPerTile = expl.getLinesPerTile(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);

                uint64_t startLine = locL + locW * linesPerTile;
                uint64_t endLine = startLine + linesPerTile - 1 + F0 - 1;

                std::vector<std::string> locLines;
                uint64_t startTile = startLine / linesPerTile;
                uint64_t endTile = endLine / linesPerTile;

                llvm::outs() << "StartLine " << startLine << ", startTile: " << startTile << "\n";
                llvm::outs() << "EndLIne: " << endLine << ", endTile: " << endLine / linesPerTile << "\n";

                for(uint64_t i = startTile; i <= endTile; i++) {
                    locLines.push_back("tile" + std::to_string(i) + "C" + std::to_string(locCa));
                }

                return locLines;
            }

            std::vector<std::string> wantLoc(AbsOpWrapper* absOp, DataflowExplorer &expl) {
                Operation* op = absOp->getUnderlyingOperation();

                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                //unsigned int locP = getAttrOrDefault(op, "locP", 0);

                std::string layerName = op->getAttr("layer_name").dyn_cast<StringAttr>().getValue().str();

                uint64_t F0 = absOp->getF0();
                // TODO fix for F1

                uint64_t linesPerTile = expl.getLinesPerTile(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);

                //llvm::outs() << "LinesPertile:  " << linesPerTile << "\n";

                unsigned int W = this->layerNameToParams[layerName].W;

                uint64_t startLine = locL + locW * linesPerTile;
                uint64_t endLine = startLine + linesPerTile - 1 + F0 - 1;

                uint64_t nStartLine = startLine + W * linesPerTile;
                uint64_t nEndLine = endLine + W * linesPerTile;

                uint64_t endLineTile = endLine / linesPerTile;
                uint64_t nStartLineTile = nStartLine / linesPerTile;
                uint64_t nEndLineTile = nEndLine / linesPerTile;

                //llvm::outs() << "startLine = " << startLine << ", endLine: " << endLine << "\n";
                llvm::outs() << "start = " << startLine / linesPerTile << ", end: " << endLine / linesPerTile << "\n";
                llvm::outs() << "nStart = " << nStartLineTile << ", nEnd: " << nEndLineTile << "\n";

                std::vector<std::string> wantLines;
                for(uint64_t i = std::max(endLineTile+1, nStartLineTile); i <= nEndLineTile; i++) {
                    wantLines.push_back("tile" + std::to_string(i) + "C" + std::to_string(locCa));
                }

                return wantLines;
            }

            std::vector<std::string> wantPrev(AbsOpWrapper* absOp, DataflowExplorer &expl) {
                Operation* op = absOp->getUnderlyingOperation();
                unsigned int locCa = getAttrOrDefault(op, "locCa", 0);
                unsigned int locL = getAttrOrDefault(op, "locL", 0);
                unsigned int locW = getAttrOrDefault(op, "locW", 0);
                //unsigned int locP = getAttrOrDefault(op, "locP", 0);

                std::string layerName = op->getAttr("layer_name").dyn_cast<StringAttr>().getValue().str();
                unsigned int W = this->layerNameToParams[layerName].W;
                unsigned int L = this->layerNameToParams[layerName].L;

                unsigned int WPrev;
                if(expl.layerNameToID[layerName] == 0) {
                    WPrev = W;
                } else {
                    WPrev = this->layerNameToParams[expl.layerIdToName[expl.layerNameToID[layerName]-1]].W;
                }

                uint64_t F0 = absOp->getF0();
                // TODO fix for F1

                uint64_t linesPerTile = expl.getLinesPerTile(expl.layerNameToID[layerName], this->layerNameToParams[layerName]);

                //llvm::outs() << "LinesPertile:  " << linesPerTile << "\n";

                // TODO double check that formula
                unsigned int highestLocTile = ((W * linesPerTile - 1) + (L-1) + F0 - 1) / linesPerTile;

                //llvm::outs() << "highestLocTile:  " << highestLocTile  << "\n";


                uint64_t startLine = locL + locW * linesPerTile;
                uint64_t endLine = startLine + linesPerTile - 1 + F0 - 1;

                uint64_t nStartLine = startLine + W * linesPerTile;
                uint64_t nEndLine = endLine + W * linesPerTile;

                uint64_t endLineTile = endLine / linesPerTile;
                uint64_t nStartLineTile = nStartLine / linesPerTile;
                uint64_t nEndLineTile = nEndLine / linesPerTile;

                //llvm::outs() << "startLine = " << startLine << ", endLine: " << endLine << "\n";
                llvm::outs() << "start = " << startLine / linesPerTile << ", end: " << endLine / linesPerTile << "\n";
                llvm::outs() << "nStart = " << nStartLineTile << ", nEnd: " << nEndLineTile << "\n";

                std::vector<std::string> wantLines;
                for(uint64_t i = std::max(endLineTile+1, nStartLineTile); i <= nEndLineTile; i++) {
                    if((i - highestLocTile - 1) >= 0) {
                        unsigned int target = (i - highestLocTile - 1) % WPrev;
                        //llvm::outs() << "want At: " << target << "\n";
                        wantLines.push_back("tile" + std::to_string(target) + "C" + std::to_string(locCa));
                    }
                }

                return wantLines;
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
                                (void)o;
                                //Operation* concatArg = o.getDefiningOp();
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
                                std::string hashString = "tile" + std::to_string(concatW) + "C" + std::to_string(concatP / concatSize);
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
                                    std::string hashString = "tile" + std::to_string(splitW) + "C" + std::to_string(pLoc);
                                    producedLineToOp[hashString] = split.getResult(i);
                                }
                            }
                        } else {
                            std::string hashString = "tile" + std::to_string(locW) + "C" + std::to_string(locP);
                            llvm::outs() << "Producing: " << hashString << "\n";
                            producedLineToOp[hashString] = prevAbsOp->getUnderlyingOperation()->getResult(0);
                        }
                    }
                }

                return producedLineToOp;
            }

            // TODO for now select arbitrary line from any core that has it, might change that
            std::map<std::string, Value> findLocalTiles(std::string layerName, DataflowExplorer &expl) {
                std::map<std::string, Value> localLines;
                //ModelParams params = this->layerNameToParams[layerName];
                std::vector<AbsOpWrapper*> absOps = this->layerNameToOps[layerName];
                std::vector<AbsOpWrapper*> toDelete;

                for(uint64_t i = 0; i < absOps.size(); i++) {
                    printOperationLoc(this->layerNameToOps[layerName].at(i)->getUnderlyingOperation());
                    std::vector<std::string> linesLoc = this->workOn(this->layerNameToOps[layerName].at(i), expl);

                    for(std::string s : linesLoc) {
                        AbsOpWrapper* absOp = this->layerNameToOps[layerName].at(i);

                        llvm::outs() << "locTiles: " << s << "\n";

                        if(absOp->getUnderlyingOperation()->getNumResults() == 2) {
                            localLines[s] = absOp->getUnderlyingOperation()->getResult(1);
                        } else {
                            unsigned int locW = getAttrOrDefault(absOp->getUnderlyingOperation(), "locW", 0);
                            mlir::torch::Torch::BaseTensorType partialRes = absOp->getUnderlyingOperation()->getResult(0).getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
                            mlir::torch::Torch::BaseTensorType forwardRes = absOp->getInput().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();

                            OpBuilder builder(absOp->getUnderlyingOperation());

                            absOp->getUnderlyingOperation()->print(llvm::outs());
                            llvm::outs() << "\n";

                            Operation* nOp = absOp->wCopy(builder, locW, std::optional<TypeRange>(TypeRange{partialRes, forwardRes}));

                            absOp->getUnderlyingOperation()->getResult(0).replaceAllUsesWith(nOp->getResult(0));

                            absOp->getUnderlyingOperation()->erase();
                            toDelete.push_back(absOp);

                            this->layerNameToOps[layerName].at(i) = opToWrapper(nOp);

                            localLines[s] = nOp->getResult(1);
                        }
                    }
                }

                for(uint64_t i = 0; i < toDelete.size(); i++) {
                    delete toDelete.at(i);
                }
                toDelete.clear();

                llvm::outs() << "Done\n";

                return localLines;
            }

            void wDuplicate(std::string layerName, unsigned int into) {
                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];

                for(int64_t i = into-1; i >= 0; i--) {
                    llvm::outs() << "IntoLoc: " << i << "\n";
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
                            Operation* op = absOp->wCopy(builder, i, std::optional<TypeRange>());

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

                                std::string hashString = "tile" + std::to_string(concatW) + "C" + std::to_string(concatP);
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

                    llvm::outs() << "Generated stuff now re-wire..\n";

                    // Re-wire duplicated one with inputs from same W group
                    std::map<std::string, AbsOpWrapper*>::iterator it;
                    for(it = paramsToLayer.begin(); it != paramsToLayer.end(); it++) {
                        AbsOpWrapper* absOp = it->second;
                        Operation* op = absOp->getUnderlyingOperation();

                        unsigned int locCa = getAttrOrDefault(absOp->getUnderlyingOperation(), "locCa", 0);
                        unsigned int locL = getAttrOrDefault(absOp->getUnderlyingOperation(), "locL", 0);
                        unsigned int locP = getAttrOrDefault(absOp->getUnderlyingOperation(), "locP", 0);

                        //unsigned int Ca = this->layerNameToParams[layerName].Ca;
                        //unsigned int P = this->layerNameToParams[layerName].P;
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

                    llvm::outs() << "And finally instantiate the concats\n";

                    // Instantiate the duplicated concats
                    std::map<std::string, std::vector<Value>>:: iterator concatIt;
                    for(concatIt = concatLocToArg.begin(); concatIt != concatLocToArg.end(); concatIt++) {
                        insertConcat(builder, concatLocToRes[it->first], concatIt->second, C_LOC, true);
                    }
                }

                // Assign new layer
                layerNameToOps[layerName] = layerOps;
            }

            void reWire(std::string layerName, DataflowExplorer &expl) {
                //OpBuilder builder(layerNameToOps[layerName].at(0)->getUnderlyingOperation());

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
                                    std::string hashString = "tile" + std::to_string(w) + "C" + std::to_string(p);
                                    std::string dupHashString = "tile" + std::to_string(w + i * ratio) + "C" + std::to_string(p);

                                    producedTiles[dupHashString] = producedTiles[hashString];
                                }
                            }
                        }
                    } else if(paramsCurr.W < paramsPrev.W) { // Insert concat on parallel tiles produced
                        unsigned int ratio = ceil((float)paramsPrev.W / paramsCurr.W);

                        llvm::outs() << "Ratio: " << ratio << "\n";

                        std::vector<Value> concats;
                        std::vector<Value> concatsArgs;

                        OpBuilder builder(this->layerNameToOps[layerName].at(0)->getUnderlyingOperation());
                        for(unsigned int p = 0; p < paramsPrev.P; p++) {
                            for(unsigned int i = 0; i < paramsPrev.W; i++) {
                                if((i != 0) && ((i % ratio) == 0)) {
                                    //OpBuilder builder(concatsArgs.at(concatsArgs.size()-1).getDefiningOp());
                                    concats.push_back(insertConcat(builder, concatsArgs.at(0), concatsArgs, N_LOC, false)->getResult(0));
                                    concatsArgs.clear();
                                } else {
                                    std::string hashString = "tile" + std::to_string(i) + "C" + std::to_string(p);
                                    llvm::outs() << "PushBack: " << hashString << "\n";
                                    concatsArgs.push_back(producedTiles[hashString]);
                                }
                            }

                            if(concatsArgs.size() != 0) {
                                //OpBuilder builder(concatsArgs.at(concatsArgs.size()-1).getDefiningOp());
                                concats.push_back(insertConcat(builder, concatsArgs.at(0), concatsArgs, N_LOC, false)->getResult(0));
                                concatsArgs.clear();
                            }

                            for(unsigned int i = 0; i < concats.size(); i++) {
                                std::string hashString = "tile" + std::to_string(i) + "C" + std::to_string(p);
                                llvm::outs() << "Keep: "  << hashString << "\n";
                                producedTiles[hashString] = concats.at(i);
                            }

                            for(unsigned int i = concats.size(); i < paramsPrev.W; i++) {
                                std::string hashString = "tile" + std::to_string(i) + "C" + std::to_string(p);
                                llvm::outs() << "remove: "  << hashString << "\n";
                                producedTiles.erase(hashString);
                            }
                        }
                    }

                    // TODO remove potential Wconcat stuff that was there before
                    // TODO or leave it to a potential clean pass
                }

                std::map<std::string, Value> locTiles = this->findLocalTiles(layerName, expl);

                llvm::outs() << "\n\nReplacing things..\n\n";

                // really re-wire from reconstructed info
                // Now easy because guarantee to find exactly what we need
                for(AbsOpWrapper* absOp : this->layerNameToOps[layerName]) {
                    Operation* op = absOp->getUnderlyingOperation();

                    printOperationLoc(op);
                    if(firstLayer) {
                        // TODO link to input of network
                    } else {
                        std::vector<std::string> wantLoc = this->wantLoc(absOp, expl);
                        llvm::outs() << "WantLoSize: " << wantLoc.size() << "\n";

                        std::vector<Value> ins;

                        for(std::string s : wantLoc) {
                            llvm::outs() << "wantLoc: " << s << "\n";
                            if(locTiles.find(s) != locTiles.end()) {
                                llvm::outs() << "wantLoc found locally: " << s << "\n";
                                //locTiles[s].print(llvm::outs());
                                assert(absOp->getInput() != Value());
                                assert(locTiles[s] != Value());

                                llvm::outs() << "Replacing: ";
                                absOp->getInput().print(llvm::outs());
                                llvm::outs() << "\n with ";
                                locTiles[s].print(llvm::outs());
                                llvm::outs() << "\n";

                                if(std::find(ins.begin(), ins.end(), locTiles[s]) == ins.end()) {
                                    ins.push_back(locTiles[s]);
                                }

                                //op->replaceUsesOfWith(absOp->getInput(), locTiles[s]);
                            }
                        }

                        std::vector<std::string> wantPrev = this->wantPrev(absOp, expl);
                        llvm::outs() << "WantPrevSize: " << wantPrev.size() << "\n";
                        for(std::string s : wantPrev) {
                            llvm::outs() << "wantPrev: " << s << "\n";
                            if(std::find(ins.begin(), ins.end(), producedTiles[s]) == ins.end()) {
                                ins.push_back(producedTiles[s]);
                            }

                            //op->replaceUsesOfWith(absOp->getInput(), producedTiles[s]);
                        }

                        assert(ins.size() >= 1);
                        if(ins.size() > 1) {
                            OpBuilder builder(op);
                            Operation* concatOp = insertConcat(builder, ins.at(0), ins, N_LOC, false);
                            op->replaceUsesOfWith(absOp->getInput(), concatOp->getResult(0));
                        } else {
                            op->replaceUsesOfWith(absOp->getInput(), ins.at(0));
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

                llvm::outs() << "wDuplicate\n";

                // duplicate graph into times
                wDuplicate(layerName, into);

                llvm::outs() << "reWrire\n";

                // Re-wire
                reWire(layerName, expl);

                llvm::outs() << "reWrire 2 \n\n\n";

                if(expl.layerNameToID[layerName] != (expl.layerNameToID.size()-1)) {
                    reWire(expl.layerIdToName[expl.layerNameToID[layerName]+1], expl);
                }

                return success();
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<func::FuncOp>("forward");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Cant find graph func\n");
                    signalPassFailure();
                    return;
                }

                DataflowExplorer dataflowExplorer = initializeLayerNameToOps(graph);
                //initializeLayerNameToParams(graph);

                // Explore topology space
                llvm::outs() << "Total Compute is: " << dataflowExplorer.getTotalCompute() << "\n";
                dataflowExplorer.enumerate();
                //dataflowExplorer.printValidTopologies();
                dataflowExplorer.dumpValidTopologies();
                dataflowExplorer.dumpParetoFrontiers();
                dataflowExplorer.dumpPathsFrom(dataflowExplorer.paretoThroughput, "./output/throughput");
                dataflowExplorer.dumpPathsFrom(dataflowExplorer.paretoLatency, "./output/latency");

                llvm::outs() << "MaxThroughput...\n";

                this->layerNameToParams = dataflowExplorer.getMaxThroughput();

                llvm::outs() << "Running expansion...\n";

                // Expand P, Ca, L for all layers
                std::map<std::string, ModelParams>::iterator it;
                unsigned int i = 0;
                for(it = layerNameToParams.begin(); it != layerNameToParams.end(); it++) {
                    unsigned int P = it->second.P;
                    unsigned int Ca = it->second.Ca;
                    unsigned int L = it->second.L;

                    llvm::outs() << "P\n";

                    if(!PTransform(it->first, P).succeeded()) {
                        llvm::outs() << "Failed to apply PTransform\n";
                        exit(1);
                    }

                    llvm::outs() << "Ca\n";

                    if(!CaTransform(it->first, Ca).succeeded()) {
                        llvm::outs() << "Failed to apply CaTransform\n";
                        exit(1);
                    }

                    llvm::outs() << "L\n";

                    if(!LTransform(it->first, L).succeeded()) {
                        llvm::outs() << "Failed to apply LTransform\n";
                        exit(1);
                    }

                    i += 1;
                }

                llvm::outs() << "W;;;\n";

                // And then W
                /*for(it = layerNameToParams.begin(); it != layerNameToParams.end(); it++) {
                    unsigned int W = it->second.W;
                    if(!WTransform(it->first, W, dataflowExplorer).succeeded()) {
                        llvm::outs() << "Failed to apply WTransform\n";
                        exit(1);
                    }
                    }*/

                /*this->layerNameToParams["conv2d_relu0"] = ModelParams(1,1,1,1,false);
                this->layerNameToParams["conv2d_relu1"] = ModelParams(1,1,3,3,true);
                this->layerNameToParams["conv2d_relu2"] = ModelParams(1,1,3,1,true);

                this->layerNameToParams["max_pool2d_with_indices0"] = ModelParams(1,1,1,1,false);
                this->layerNameToParams["max_pool2d_with_indices1"] = ModelParams(1,1,1,1,false);

                //PTransform("conv2d_relu1", 4);
                //PTransform("max_pool2d_with_indices1", 4);
                //CaTransform("conv2d_relu2", 4);

                LTransform("conv2d_relu1", 3);
                LTransform("conv2d_relu2", 3);

                llvm::outs() << "WTransform\n";

                // W expand
                WTransform("conv2d_relu1", 3, dataflowExplorer);*/

                llvm::outs() << "Cleaning..\n";

                clearLayerNameToOps();

                //exit(1);
            }
        };
    }
}

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createXTenDataflowPass() {
    return std::make_unique<XTenDataflowPass>();
}

} // namespace xten
} // namespace xilinx
