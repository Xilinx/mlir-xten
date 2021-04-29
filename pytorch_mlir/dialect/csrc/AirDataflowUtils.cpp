#include "AirDataflowUtils.h"
#include "mlir/IR/OperationSupport.h"


#define DEBUG_TYPE "air-dataflow-utils"

using namespace mlir;

// Weight locations
#define COUT_LOC 0
#define CIN_LOC 1
#define F0_LOC 2
#define F1_LOC 3

// Acts locs
#define C_LOC 0
#define N_LOC 1
#define M_LOC 2

namespace xilinx {
    namespace air {
        ShapedType halveShapeAt(ShapedType initShape, unsigned int at) {
            auto shape = initShape.getShape();
            std::vector<long> newShape = std::vector<long>(shape);
            newShape[at] = newShape[at] / 2;
            shape = initShape.getShape();
            //int i = 0;
            //for(auto e : shape) {
            //llvm::outs() << "Got shape: " << e << " vs " << newShape[i] << "\n";
            //newShape.push_back(e);
            //i++;
            //}

            ArrayRef<long> nShape = ArrayRef<long>(newShape);
            ShapedType ttype = RankedTensorType::get(nShape, initShape.getElementType());

            return ttype;
        }

        // TODO most likely factor some code here
        void splitConstantActivations(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, unsigned int loc, DenseElementsAttr at) {
            ShapedType initialShape = at.getType();
            ArrayRef<int64_t> s = initialShape.getShape();

            uint64_t C = s[CIN_LOC];
            uint64_t N = s[N_LOC];
            uint64_t M = s[M_LOC];

            uint64_t C_switch = C;
            uint64_t N_switch = N;
            uint64_t M_switch = M;

            if(loc == 0) {
                C_switch = C / 2;
            } else if(loc == 1) {
                N_switch = N / 2;
            } else if(loc == 2) {
                M_switch = M / 2;
            }

            if(initialShape.getElementType().isF32()) { // TODO more types
                std::vector<std::vector<APFloat>> vects;

                vects.push_back(std::vector<APFloat>());
                vects.push_back(std::vector<APFloat>());

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    //llvm::outs() << "Got this value: ";
                    //(*it).print(llvm::outs());
                    uint64_t loc_c = i / (M * N);
                    uint64_t loc_N = (i / M) % N;
                    uint64_t loc_M = i % M;

                    uint64_t vectsId = std::max(std::max(loc_c / C_switch, loc_N / N_switch),
                                                loc_M / M_switch);

                    vects.at(vectsId).push_back(*it);
                    i++;
                }

                for(uint64_t i = 0; i < vects.size(); i++) {
                    assert(vects.at(i).size() == (at.getType().getNumElements() / vects.size()));
                }

                ShapedType ttype = halveShapeAt(initialShape, loc);

                for(uint64_t i = 0; i < vects.size(); i++) {
                    DenseElementsAttr attr = DenseElementsAttr::get(ttype, vects.at(i));
                    Operation* cst = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attr);
                    ops.push_back(cst->getResult(0));
                }
            }
        }

        // Splits half for the Cin / Cout dim and in all for F dims
        void splitConstantWeights(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, unsigned int loc, DenseElementsAttr at) {
            ShapedType initialShape = at.getType();
            ArrayRef<int64_t> s = initialShape.getShape();

            uint64_t COut = s[COUT_LOC];
            uint64_t CIn = s[CIN_LOC];
            uint64_t F0 = s[F0_LOC];
            uint64_t F1 = s[F1_LOC];

            uint64_t COut_switch = COut;
            uint64_t CIn_switch = CIn;
            uint64_t F0_switch = F0;
            uint64_t F1_switch = F1;

            if(loc == 0) {
                COut_switch = COut / 2;
            } else if(loc == 1) {
                CIn_switch = CIn / 2;
            } else if(loc == 2) {
                F0_switch = 1;
            } else if(loc == 3) {
                F1_switch = 1;
            }

            uint64_t i = 0;
            if(initialShape.getElementType().isF32()) { // TODO is this the only choice?
                std::vector<std::vector<APFloat>> vects;

                if(loc == 0) {
                    vects.push_back(std::vector<APFloat>());
                    vects.push_back(std::vector<APFloat>());
                } else if(loc == 1) {
                    vects.push_back(std::vector<APFloat>());
                    vects.push_back(std::vector<APFloat>());
                } else if(loc == 2) {
                    for(uint64_t i = 0; i < F1; i++) {
                        vects.push_back(std::vector<APFloat>());
                    }
                } else if(loc == 3) {
                    for(uint64_t i = 0; i < F1; i++) {
                        vects.push_back(std::vector<APFloat>());
                    }
                }

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    //llvm::outs() << "Got this value: ";
                    //(*it).print(llvm::outs());

                    uint64_t loc_cout = i / (F0 * F1 * CIn);
                    uint64_t loc_cin = (i / (F0 * F1)) % CIn;
                    uint64_t loc_F0 = (i / F1) % F0;
                    uint64_t loc_F1 = i % F1;

                    uint64_t vectsId = std::max(std::max(loc_cout / COut_switch, loc_cin / CIn_switch),
                                                std::max(loc_F0 / F0_switch, loc_F1 / F1_switch));

                    vects.at(vectsId).push_back(*it);
                    i++;
                }

                for(uint64_t i = 0; i < vects.size(); i++) {
                    assert(vects.at(i).size() == (at.getType().getNumElements() / vects.size()));
                }

                ShapedType ttype = halveShapeAt(initialShape, loc);

                for(uint64_t i = 0; i < vects.size(); i++) {
                    DenseElementsAttr attr = DenseElementsAttr::get(ttype, vects.at(i));
                    Operation* cst = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attr);
                    ops.push_back(cst->getResult(0));
                }
            }
        }

        // loc = 0 split
        // loc > 0 generate some other 0 biases
        void splitConstantBias(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, unsigned int loc, DenseElementsAttr at) {
            ShapedType initialShape = at.getType();
            if(initialShape.getElementType().isF32()) { // TODO extend to more types
                std::vector<std::vector<APFloat>> vects;

                vects.push_back(std::vector<APFloat>());
                vects.push_back(std::vector<APFloat>());

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    if(loc == 0) {
                        if(i < (at.getType().getNumElements() / 2)) {
                            vects.at(0).push_back(*it);
                        } else {
                            vects.at(1).push_back(*it);
                        }
                    } else {
                        vects.at(0).push_back(*it);
                        // NOTE assume that same kernel with 0 bias from the compiler point of view
                        vects.at(1).push_back(APFloat((float)0));
                    }
                    i++;
                }

                // assert((split == PSplit) && (left.size() == right.size() == (at.getType().getNumElements() / 2)) ||
                //        (split == CaSplit) && (left.size() == right.size() == at.getType().getNumElements()));

                // now splitted the dense in two part, need to regenerate it
                ShapedType ttype;
                if(loc == 0) {
                    ttype = halveShapeAt(initialShape, 0);
                } else {
                    ttype = initialShape;
                }

                for(uint64_t i = 0; i < vects.size(); i++) {
                    DenseElementsAttr attr = DenseElementsAttr::get(ttype, vects.at(i));
                    Operation* cst = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attr);
                    ops.push_back(cst->getResult(0));
                }
            }
        }

        // TODO support WSplit
        unsigned int splitToDim(Split split, SplitType t) {
            if(t == bSplitType) {
                if(split == PSplit) {
                    return 0;
                } else {
                    return 1;
                }
            } else if(t == aSplitType) {
                if(split == CaSplit) {
                    return 0;
                } else {
                    return (unsigned int )-1;
                }
            } else if(t == wSplitType) {
                if(split == PSplit) {
                    return 0;
                } else if(split == CaSplit) {
                    return 1;
                } else if(split == LSplit) {
                    return 2;
                }
            }
        }

        void splitConstant(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, SplitType t) {
            for(NamedAttribute attr: op->getAttrs()) {
                // We Look for the Dense Attribute
                auto at = attr.second.dyn_cast<DenseElementsAttr>();
                if(at) {
                    if(t == bSplitType) {
                        unsigned int splitDim = splitToDim(split, t);
                        splitConstantBias(op, ops, builder, splitDim, at);
                    } else if(t == aSplitType) {
                        unsigned int splitDim = splitToDim(split, t);
                        if(splitDim == (unsigned int)-1) {
                            // TODO maybe fail silently if top level is fine with that
                            llvm::outs() << "Only Ca split is supported to split activation tensors";
                            exit(1);
                        } else {
                            splitConstantActivations(op, ops, builder, splitDim, at);
                        }
                    } else {
                        unsigned int splitDim = splitToDim(split, t);
                        splitConstantWeights(op, ops, builder, splitDim, at);
                    }

                }
            }
        }
    }
}

