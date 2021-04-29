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
#include <iostream>
#include <vector>
#include <set>

#define DEBUG_TYPE "air-dataflow-pass"

using namespace mlir;

// TODO make the patterns generic
//   - Check with concat etc..
//   - Insert concat / split whenever needed
//   - Restriction from architectural constraints

namespace xilinx {
    namespace air {

        struct PPattern : public OpRewritePattern<Conv2dReLUOp> {
        public:
            PPattern(MLIRContext *context) : OpRewritePattern<Conv2dReLUOp>(context, 1) {}

            LogicalResult matchAndRewrite(Conv2dReLUOp op, PatternRewriter &rewriter) const override {
                OpBuilder builder(op.getOperation());
                std::vector<Value> nConsts;
                std::vector<Value> nBiases;

                // Split weights
                Operation* weights = op.weight().getDefiningOp();//->getName();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                    splitConstant(constOp, nConsts, builder, PSplit, wSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split biases
                Operation* biases = op.bias().getDefiningOp();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                    splitConstant(constOp, nBiases, builder, PSplit, bSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split Return Type shape
                ShapedType nReturnType = halveShapeAt(op.getResult().getType().dyn_cast<ShapedType>(), 1);
                llvm::outs() << "Return Type: " << op.getResult().getType() << " and new is " << nReturnType << "\n";

                // Generate new convs
                Operation* lconv = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                                nReturnType,
                                                                op.input(),
                                                                nConsts.at(0),
                                                                nBiases.at(0),
                                                                op.stride(),
                                                                op.padding(),
                                                                op.dilation(),
                                                                op.transposed(),
                                                                op.output_padding(),
                                                                op.groups());

                Operation* rconv = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                                nReturnType,
                                                                op.input(),
                                                                nConsts.at(1),
                                                                nBiases.at(1),
                                                                op.stride(),
                                                                op.padding(),
                                                                op.dilation(),
                                                                op.transposed(),
                                                                op.output_padding(),
                                                                op.groups());

                // Concat
                ShapedType concatResType = op.getResult().getType().dyn_cast<ShapedType>();

                std::vector<Value> vec;
                vec.push_back(lconv->getResult(0));
                vec.push_back(rconv->getResult(0));
                ArrayRef<Value> convsRef = ArrayRef<Value>(vec);
                ValueRange convs(convsRef);

                // TODO check the width here
                Operation* cstDim = builder.create<ConstantIntOp>(builder.getUnknownLoc(), 0, 32);

                Operation* res = builder.create<ConcatOp>(builder.getUnknownLoc(), concatResType, convs, cstDim->getResult(0));

                // Replace output of old convolution usage by concat value
                op.getResult().replaceAllUsesWith(res->getResult(0));

                // Delete previous Csts and ConvolutionOp
                weights->erase();
                biases->erase();
                op.erase();

                return success();
            }
        };

        struct CaPattern : public OpRewritePattern<Conv2dReLUOp> {
        public:
            CaPattern(MLIRContext *context) : OpRewritePattern<Conv2dReLUOp>(context, 1) {}

            LogicalResult matchAndRewrite(Conv2dReLUOp op, PatternRewriter &rewriter) const override {
                OpBuilder builder(op.getOperation());
                std::vector<Value> nConsts;
                std::vector<Value> nBiases;
                std::vector<Value> nActivations;

                // Split weights
                Operation* weights = op.weight().getDefiningOp();//->getName();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                    splitConstant(constOp, nConsts, builder, CaSplit, wSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split biases
                Operation* biases = op.bias().getDefiningOp();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                    splitConstant(constOp, nBiases, builder, CaSplit, bSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // split activations
                if(auto constOp = op.input().getDefiningOp<ConstantOp>()) {
                    splitConstant(constOp, nActivations, builder, CaSplit, aSplitType);
                } else {
                    llvm::outs() << "Inserting a split operation!\n";
                    // This is then a normal value, break it down
                    ShapedType shape = halveShapeAt(op.input().getType().dyn_cast<ShapedType>(), 1);
                    //std::vector<ShapedType> shapes = ArrayRef<ShapedType>(std::vector<ShapedType>{shape, shape});

                    Operation* cstDim = builder.create<ConstantIntOp>(builder.getUnknownLoc(), 1, 32);
                    Operation* splitOp = builder.create<SplitOp>(builder.getUnknownLoc(), TypeRange({shape, shape}), op.input(), cstDim->getResult(0));

                    nActivations.push_back(splitOp->getResult(0));
                    nActivations.push_back(splitOp->getResult(1));
                }

                // Split Return Type shape
                ShapedType nReturnType = halveShapeAt(op.getResult().getType().dyn_cast<ShapedType>(), 1);
                llvm::outs() << "Return Type: " << op.getResult().getType() << " and new is " << nReturnType << "\n";

                // Generate new convs
                Operation* lconv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                       nReturnType,
                                                                       nActivations.at(0),
                                                                       nullptr,
                                                                       nConsts.at(0),
                                                                       nBiases.at(0),
                                                                       op.stride(),
                                                                       op.padding(),
                                                                       op.dilation(),
                                                                       op.transposed(),
                                                                       op.output_padding(),
                                                                       op.groups());

                Operation* rconv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                       op.getResult().getType(),
                                                                       nActivations.at(1),
                                                                       lconv->getResult(0),
                                                                       nConsts.at(1),
                                                                       nBiases.at(1),
                                                                       op.stride(),
                                                                       op.padding(),
                                                                       op.dilation(),
                                                                       op.transposed(),
                                                                       op.output_padding(),
                                                                       op.groups());

                // Replace output of old convolution usage by concat value
                op.getResult().replaceAllUsesWith(rconv->getResult(0));

                // Delete previous Csts and ConvolutionOp
                weights->erase();
                biases->erase();
                op.erase();

                return success();
            }
        };

        struct LPattern : public OpRewritePattern<Conv2dReLUOp> {
        public:
            LPattern(MLIRContext *context) : OpRewritePattern<Conv2dReLUOp>(context, 1) {}

            LogicalResult matchAndRewrite(Conv2dReLUOp op, PatternRewriter &rewriter) const override {
                OpBuilder builder(op.getOperation());
                std::vector<Value> nConsts;
                std::vector<Value> nBiases;

                // Split weights
                Operation* weights = op.weight().getDefiningOp();//->getName();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                    splitConstant(constOp, nConsts, builder, LSplit, wSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split biases
                Operation* biases = op.bias().getDefiningOp();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                    splitConstant(constOp, nBiases, builder, LSplit, bSplitType);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }


                // Split Return Type shape
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

                uint64_t F = nConsts.size();
                for(unsigned int i = 1; i < F; i++) {
                    nConv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                (i == (F-1)) ? TypeRange({returnType}) : TypeRange({returnType, returnType}),
                                                                forward,
                                                                partial,
                                                                nConsts.at(i),
                                                                nBiases.at(1),
                                                                op.stride(),
                                                                op.padding(),
                                                                op.dilation(),
                                                                op.transposed(),
                                                                op.output_padding(),
                                                                op.groups());
                    partial = nConv->getResult(0);
                    if(i < (F-1)) {
                        forward = nConv->getResult(1);
                    }

                }


                // Replace output of old convolution usage by concat value
                op.getResult().replaceAllUsesWith(nConv->getResult(0));

                // Delete previous Csts and ConvolutionOp
                weights->erase();
                biases->erase();
                op.erase();

                return success();
            }
        };

        class MyPatternRewriter : public PatternRewriter {
        public:
            MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

        };

        struct AirDataflowPass : public PassWrapper<AirDataflowPass, OperationPass<ModuleOp>> {
        private:
            std::map<std::string, std::vector<Operation*>> layerNameToOps;

        public:
            AirDataflowPass() {}

            void runOnOperation() override {
                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<FuncOp>("graph");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Cant find graph func\n");
                    signalPassFailure();
                    return;
                }

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

                // Take context from the top level Module?
                PPattern p(module->getContext());
                CaPattern ca(module->getContext());
                LPattern l(module->getContext());
                MyPatternRewriter rewriter(module->getContext());

                // expand slowest layer
                graph.walk([&](Operation *op) {
                        Conv2dReLUOp conv = llvm::dyn_cast<Conv2dReLUOp>(op);
                        if(conv) {
                            l.matchAndRewrite(conv, rewriter);
                        }
                    });
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

