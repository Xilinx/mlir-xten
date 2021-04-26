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

#define DEBUG_TYPE "aten-test-rewrite"

using namespace mlir;

namespace xilinx {
    namespace air {

        // TODO make the patterns generic and tests
        struct PPattern : public OpRewritePattern<Conv2dReLUOp> {
        public:
            PPattern(MLIRContext *context) : OpRewritePattern<Conv2dReLUOp>(context, 1) {}

            LogicalResult matchAndRewrite(Conv2dReLUOp op, PatternRewriter &rewriter) const override {
                OpBuilder builder(op.getOperation());
                std::vector<Operation*> nConsts;
                std::vector<Operation*> nConvs;
                std::vector<Operation*> nBiases;

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
                                                                nConsts.at(0)->getResult(0),
                                                                nBiases.at(0)->getResult(0),
                                                                op.stride(),
                                                                op.padding(),
                                                                op.dilation(),
                                                                op.transposed(),
                                                                op.output_padding(),
                                                                op.groups());

                Operation* rconv = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                                nReturnType,
                                                                op.input(),
                                                                nConsts.at(1)->getResult(0),
                                                                nBiases.at(1)->getResult(0),
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
                std::vector<Operation*> nConsts;
                std::vector<Operation*> nConvs;
                std::vector<Operation*> nBiases;

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

                // Split Return Type shape
                ShapedType nReturnType = halveShapeAt(op.getResult().getType().dyn_cast<ShapedType>(), 1);
                llvm::outs() << "Return Type: " << op.getResult().getType() << " and new is " << nReturnType << "\n";

                // Start from Conv so no partialIn for first; and no partial out for second
                auto optAttrIn = builder.getI64ArrayAttr(ArrayRef<int64_t>(std::vector<int64_t>(0, 1)));
                auto optAttrOut = builder.getI64ArrayAttr(ArrayRef<int64_t>(std::vector<int64_t>(1, 0)));

                // Generate new convs
                Operation* lconv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                       nReturnType,
                                                                       op.input(),
                                                                       nullptr,
                                                                       nConsts.at(0)->getResult(0),
                                                                       nBiases.at(0)->getResult(0),
                                                                       op.stride(),
                                                                       op.padding(),
                                                                       op.dilation(),
                                                                       op.transposed(),
                                                                       op.output_padding(),
                                                                       op.groups());

                Operation* rconv = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                       nReturnType,
                                                                       op.input(),
                                                                       lconv->getResult(0),
                                                                       nConsts.at(1)->getResult(0),
                                                                       nBiases.at(1)->getResult(0),
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

        class MyPatternRewriter : public PatternRewriter {
        private:
            std::set<Operation *> toDelete;

        public:
            MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

        };

        struct AirDataflowPass : public PassWrapper<AirDataflowPass, OperationPass<ModuleOp>> {
        private:
            // TODO map layerName -> operations

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

                // Take context from the top level Module?
                CaPattern p(module->getContext());
                MyPatternRewriter rewriter(module->getContext());

                graph.walk([&](Operation *op) {
                        Conv2dReLUOp conv = llvm::dyn_cast<Conv2dReLUOp>(op);
                        if(conv) {
                            p.matchAndRewrite(conv, rewriter);
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

