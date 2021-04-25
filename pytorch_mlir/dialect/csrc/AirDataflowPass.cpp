// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AirDataflow.h"
#include "AIRDialect.h"

#include <iostream>
#include <vector>
#include <set>

#define DEBUG_TYPE "aten-test-rewrite"

using namespace mlir;

// TODO Try to generate Conv2dRelu
// TODO Why OpRewritePattern specifically and not any of the similar ones?
// TODO This might work on convolution, but might want to write something more generic afterwards

namespace xilinx {
    namespace air {
        struct PPattern : public OpRewritePattern<Conv2dReLUOp> {
        public:
            // NOTE benefit = 1?
            PPattern(MLIRContext *context) : OpRewritePattern<Conv2dReLUOp>(context, 1) {}

            ShapedType halveShapeAt(ShapedType initShape, unsigned int at) const {
                auto shape = initShape.getShape();
                std::vector<long> newShape = std::vector<long>(shape);
                newShape[at] = newShape[at] / 2;
                shape = initShape.getShape();
                int i = 0;
                for(auto e : shape) {
                    llvm::outs() << "Got shape: " << e << " vs " << newShape[i] << "\n";
                    //newShape.push_back(e);
                    i++;
                }

                ArrayRef<long> nShape = ArrayRef<long>(newShape);
                ShapedType ttype = RankedTensorType::get(nShape, initShape.getElementType());

                return ttype;
            }

            void splitConstant(ConstantOp op, std::vector<Operation*> &ops, OpBuilder &builder) const {
                llvm::outs() << op->getAttrs().size() << " attributes;\n";

                for(NamedAttribute attr: op->getAttrs()) {
                    // We only have a Dense attribute
                    auto at = attr.second.dyn_cast<DenseElementsAttr>();

                    if(at) {
                        ShapedType initialShape = at.getType();
                        // TODO how to use correct types here?
                        if(initialShape.getElementType().isF32()) {
                            // Extract the two parts of the Dense Attribute

                            std::vector<APFloat> left;
                            std::vector<APFloat> right;

                            int64_t elems = at.getType().getNumElements();
                            llvm::outs() << "Has # " << elems << " in that type\n";
                            int64_t loc = 0;

                            llvm::outs() << "Type is: " << at.getType().getElementType() << "\n";
                            // TODO check traversal assumption here
                            for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                                //llvm::outs() << "Got this value: ";
                                //(*it).print(llvm::outs());

                                if(loc < (elems / 2)) {
                                    left.push_back(*it);
                                } else {
                                    right.push_back(*it);
                                }

                                loc++;
                            }

                            // now splitted the dense in two part, need to regenerate it
                            ShapedType ttype = this->halveShapeAt(initialShape, 0);

                            DenseElementsAttr attrLeft = DenseElementsAttr::get(ttype, left);
                            DenseElementsAttr attrRight = DenseElementsAttr::get(ttype, right);

                            // Construct the new constants
                            Operation* cstl = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrLeft);
                            Operation* cstr = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrRight);

                            ops.push_back(cstl);
                            ops.push_back(cstr);
                        }
                    }
                }
            }

            LogicalResult matchAndRewrite(Conv2dReLUOp op, PatternRewriter &rewriter) const override {
                OpBuilder builder(op.getOperation());
                std::vector<Operation*> nConsts;
                std::vector<Operation*> nConvs;
                std::vector<Operation*> nBiases;

                // Split weights
                Operation* weights = op.weight().getDefiningOp();//->getName();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(weights)) {
                    this->splitConstant(constOp, nConsts, builder);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split biases
                Operation* biases = op.bias().getDefiningOp();
                if(auto constOp = llvm::dyn_cast<ConstantOp>(biases)) {
                    this->splitConstant(constOp, nBiases, builder);
                } else {
                    llvm::outs() << "Cannot convert to ConstOp!\n";
                }

                // Split Return Type shape
                ShapedType nReturnType = this->halveShapeAt(op.getResult().getType().dyn_cast<ShapedType>(), 1);
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

        class MyPatternRewriter : public PatternRewriter {
        private:
            std::set<Operation *> toDelete;

        public:
            MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

        };

        struct ATenTestRewritePass : public PassWrapper<ATenTestRewritePass, OperationPass<ModuleOp>> {
        private:
            // TODO add analysis variables here?

        public:
            void runOnOperation() override {
                ModuleOp module = getOperation();

                auto graph = module.lookupSymbol<FuncOp>("graph");
                if(!graph) {
                    emitError(UnknownLoc::get(module.getContext()), "Cant find graph func\n");
                    signalPassFailure();
                    return;
                }

                // Take context from the top level Module?
                PPattern p(module->getContext());
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

// TODO rename to AIR based pass
namespace xilinx {
    namespace air {
        std::unique_ptr<mlir::Pass> createATenTestRewritePass() {
            return std::make_unique<ATenTestRewritePass>();
        }

    } // namespace aten
} // namespace xilinx

void xilinx::air::registerATenTestRewritePass() {
    PassRegistration<ATenTestRewritePass>("air-expand-graph",
                                          "Dataflow expansion of ATen NN graph towards AIE implementation");
}

