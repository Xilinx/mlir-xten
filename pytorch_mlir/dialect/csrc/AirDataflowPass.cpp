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

// TODO make the patterns generic
//   - Restriction from architectural constraints
//   - Support all convolution types

// TODO Make sure that the communication is not shuffled somehow
// TODO make sure that dim of split is correct one
// TODO write verifiers

// TODO re add the wrapper when generating stuff
// TODO make sure that the location annotation is added
// TODO Define rules to setup connections

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

                    // Split weights
                    Operation* weights;
                    if(genOp->hasWeights()) {
                        weights = genOp->getWeights().getDefiningOp();
                        if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                            splitConstantInto(constOp, nConsts, builder, PSplit, wSplitType, into);
                        } else {
                            llvm::outs() << "Cannot convert to ConstOp!\n";
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

                        if(genOp->hasWeights()) {
                            conv = genOp->buildOp(builder,
                                                  TypeRange({nReturnType}),
                                                  input,
                                                  llvm::Optional<Value>(nConsts.at(i)),
                                                  llvm::Optional<Value>(nBiases.at(i)),
                                                  llvm::Optional<Value>(), false);

                            assert(conv != nullptr);
                        } else {
                            conv = genOp->buildOp(builder,
                                                  TypeRange({nReturnType}),
                                                  input,
                                                  llvm::Optional<Value>(),
                                                  llvm::Optional<Value>(),
                                                  llvm::Optional<Value>(), false);

                            assert(conv != nullptr);
                        }

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
                        Operation* nConv = genOp->buildOp(builder, TypeRange({op->getResult(0).getType()}),
                                                         nInputs.at(i), llvm::Optional<Value>(nConsts.at(i)),
                                                         llvm::Optional<Value>(nBiases.at(i)),
                                                         llvm::Optional<Value>(conv->getResult(0)), true);

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
                        splitConstantInto(constOp, nBiases, builder, LSplit, bSplitType, into);
                    } else {
                        llvm::outs() << "Cannot convert to ConstOp!\n";
                    }

                    // Same return type here
                    ShapedType retType = op->getResult(0).getType().dyn_cast<ShapedType>();

                    // Generate new convs
                    Operation* nConv = genOp->buildOp(builder, TypeRange({retType, retType}),
                                                     genOp->getInput(), llvm::Optional<Value>(nConsts.at(0)),
                                                     llvm::Optional<Value>(nBiases.at(0)), llvm::Optional<Value>(), true);

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
                        nConv = genOp->buildOp(builder,
                                               (i == (into-1)) ? TypeRange({retType}) : TypeRange({retType, retType}),
                                               forward, llvm::Optional<Value>(nConsts.at(i)),
                                               llvm::Optional<Value>(nBiases.at(i)),
                                               llvm::Optional<Value>(partial), false);

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

            void annotateLines() {
                std::map<std::string, std::vector<AbsOpWrapper*>>::iterator it;

                // set line
                for(it = this->layerNameToOps.begin(); it != this->layerNameToOps.end(); it++) {
                    for(AbsOpWrapper* absOp : it->second) {
                        Operation* op = absOp->getUnderlyingOperation();
                        OpBuilder builder(op);

                        if(op->getAttr("locL") != nullptr) { // Responsible for handling only one line
                            unsigned int locL = op->getAttr("locL").dyn_cast<IntegerAttr>().getValue().getZExtValue();
                            unsigned int L = this->layerNameToParams[it->first].L;
                            //auto ty = IntegerType::get(op->getContext(), 32);
                            //auto attr = IntegerAttr::get(ty, locL);
                            auto attr = builder.getI32ArrayAttr({static_cast<int>(L - locL - 1)});

                            op->setAttr(llvm::StringRef("line"), attr);
                        } else { // Responsible for any number (only one tile produced, default to 0)
                            //auto ty = IntegerType::get(op->getContext(), 32);
                            //auto attr = IntegerAttr::get(ty, 0);

                            unsigned int F = absOp->getKernelSize();
                            auto attr = builder.getI32ArrayAttr({0, static_cast<int>(F)-1});

                            op->setAttr(llvm::StringRef("line"), attr);
                        }
                    }
                }
            }

            unsigned int getAttrOrDefault(Operation* op, std::string attrName, unsigned int defVal) {
                if(op->getAttr(attrName) != nullptr) {
                    return op->getAttr(attrName).dyn_cast<IntegerAttr>().getValue().getZExtValue();
                } else {
                    return defVal;
                }
            }

            // TODO at the moment force work at line grannularity, need to later generalize to tile grannularity possibly
            LogicalResult WTransform(std::string layerName, unsigned int into) {
                if(into == 1) {
                    return success();
                }

                std::vector<AbsOpWrapper*> layerOps = layerNameToOps[layerName];

                // duplicate graph into times
                for(int64_t i = into-1; i >= 0; i--) {
                    OpBuilder builder(layerNameToOps[layerName].at(0)->getUnderlyingOperation());
                    std::map<std::string, AbsOpWrapper*> paramsToLayer;
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

                            unsigned int locCa = getAttrOrDefault(absOp->getUnderlyingOperation(), "locCa", 0);
                            unsigned int locL = getAttrOrDefault(absOp->getUnderlyingOperation(), "locL", 0);
                            unsigned int locP = getAttrOrDefault(absOp->getUnderlyingOperation(), "locP", 0);
                            std::string hashString = "P" + std::to_string(locP) +
                                "Ca" + std::to_string(locCa) +
                                "L" + std::to_string(locL);

                            paramsToLayer[hashString] = locAbsOp;
                        }
                    }

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
                }

                layerNameToOps[layerName] = layerOps;

                // rewire loc forward and cascades

                // set wantLine
                // TODO double check when only one want if still correct operation in following function
                std::map<std::string, std::vector<AbsOpWrapper*>>::iterator it;
                for(it = this->layerNameToOps.begin(); it != this->layerNameToOps.end(); it++) {
                    for(AbsOpWrapper* absOp : it->second) {
                        Operation* op = absOp->getUnderlyingOperation();
                        OpBuilder builder(op);

                        auto lines = op->getAttr("line").dyn_cast<ArrayAttr>().getValue();
                        if(lines.size() == 1) {
                            unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                            auto attr = builder.getI32ArrayAttr({static_cast<int>(lines0 + into)});
                            op->setAttr(llvm::StringRef("wantLine"), attr);
                        } else {
                            unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();
                            unsigned int lines1 = lines[1].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                            auto attr = builder.getI32ArrayAttr({static_cast<int>(lines0 + into), static_cast<int>(lines1 + into)});
                            op->setAttr(llvm::StringRef("wantLine"), attr);
                        }
                    }
                }

                // Re-wire

                // find lines locations from local layer
                std::map<uint64_t, std::vector<AbsOpWrapper*>> lineToOp;
                for(AbsOpWrapper* absOp : layerNameToOps[layerName]) {
                    auto lines = absOp->getUnderlyingOperation()->getAttr("line").dyn_cast<ArrayAttr>().getValue();
                    if(lines.size() == 1) {
                        unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                        lineToOp[lines0].push_back(absOp);
                    } else {
                        unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();
                        unsigned int lines1 = lines[1].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                        for(unsigned int i = lines0; i < lines1; i++) {
                            lineToOp[i].push_back(absOp);
                        }
                    }
                }

                // Find lines locations from previous layer
                std::map<std::string, AbsOpWrapper*> producedLineToOp;
                std::vector<std::string>::iterator layerLoc;
                layerLoc = std::find(this->layerOrdering.begin(), this->layerOrdering.end(), layerName);

                // preserve all mappings as if not from that layer is from NN input so we are fine if first layer
                if(layerLoc != this->layerOrdering.begin()) {
                    layerLoc--;
                    ModelParams prevParams = this->layerNameToParams[*layerLoc];
                    for(AbsOpWrapper* prevAbsOp : this->layerNameToOps[*layerLoc]) {
                        unsigned int locCa = getAttrOrDefault(prevAbsOp->getUnderlyingOperation(), "locCa", 0);
                        unsigned int locL = getAttrOrDefault(prevAbsOp->getUnderlyingOperation(), "locL", 0);
                        unsigned int locW = getAttrOrDefault(prevAbsOp->getUnderlyingOperation(), "locW", 0);
                        unsigned int locP = getAttrOrDefault(prevAbsOp->getUnderlyingOperation(), "locP", 0);

                        if(locCa == (prevParams.Ca-1) && locL == (prevParams.L-1)) { // is a producer
                            std::string hashString = "P"+ std::to_string(locP)+ "W" + std::to_string(locW);
                            llvm::outs() << "hs: " << hashString << "\n";
                            producedLineToOp[hashString] = prevAbsOp;
                        }
                    }
                }

                llvm::outs() << "Found prev lines\n";

                // really re-wire from reconstructed info
                for(AbsOpWrapper* absOp : layerOps) {
                    Operation* op = absOp->getUnderlyingOperation();
                    auto wantLines = op->getAttr("wantLine").dyn_cast<ArrayAttr>().getValue();
                    auto locW = op->getAttr("locW").dyn_cast<IntegerAttr>().getValue().getZExtValue();

                    unsigned int want0 = wantLines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();
                    unsigned int want1 = (wantLines.size() == 1) ? want0 + 1 : wantLines[1].dyn_cast<IntegerAttr>().getValue().getZExtValue();
                    for(unsigned int i = want0; i < want1; i++) {
                        if(lineToOp.find(i) == lineToOp.end()) {
                            llvm::outs() << "Found something from the previous layer...\n";
                            // Need to find producer in previous layer
                            unsigned int F = absOp->getKernelSize();
                            unsigned int locW = op->getAttr("locW").dyn_cast<IntegerAttr>().getValue().getZExtValue();
                            unsigned int locCa = getAttrOrDefault(op, "locCa", 0);

                            std::string hashString = "P" + std::to_string(locCa) + "W" + std::to_string(locW -(into + F - 1));
                            if(producedLineToOp.find(hashString) != producedLineToOp.end()) {
                                llvm::outs() << "found with same P\n";
                                AbsOpWrapper* producer = producedLineToOp[hashString];
                                op->replaceUsesOfWith(absOp->getInput(), producer->getUnderlyingOperation()->getResult(0));
                            } else {
                                llvm::outs() << "found with different P\n";
                                hashString = "P" + std::to_string(locCa) + "W0";
                                llvm::outs()<< "Querrying: " << hashString << "\n";
                                AbsOpWrapper* producer = producedLineToOp[hashString];
                                op->replaceUsesOfWith(absOp->getInput(), producer->getUnderlyingOperation()->getResult(0));
                            }
                        } else {
                            llvm::outs() << "Found something from the current layer...\n";
                            // prefer from same W group and then closest W group
                            std::vector<AbsOpWrapper*> ops = lineToOp[i];
                            unsigned int closest = (unsigned int)-1;
                            AbsOpWrapper* closestOp;
                            for(AbsOpWrapper* lineAbsOp : ops) {
                                Operation* lineOperation = lineAbsOp->getUnderlyingOperation();
                                unsigned int nLocW = lineOperation->getAttr("locW").dyn_cast<IntegerAttr>().getValue().getZExtValue();
                                unsigned int locW = op->getAttr("locW").dyn_cast<IntegerAttr>().getValue().getZExtValue();

                                unsigned int diff = (unsigned int)abs((float)nLocW - locW);
                                if(diff < closest) {
                                    closest = diff;
                                    closestOp = lineAbsOp;
                                }
                            }

                            op->replaceUsesOfWith(absOp->getInput(), closestOp->getUnderlyingOperation()->getResult(0));
                        }
                    }
                }

                // TODO remove concat ops if W prev == W curr and rewrite it otherwise

                // TODO insert concat ops when required

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
                    }*/

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
                WTransform("conv2d_relu1", 3);

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

