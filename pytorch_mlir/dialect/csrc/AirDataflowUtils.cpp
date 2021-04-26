#include "AirDataflowUtils.h"
#include "mlir/IR/OperationSupport.h"


#define DEBUG_TYPE "air-dataflow-utils"

using namespace mlir;

#define COUT_LOC 0
#define CIN_LOC 1
#define F0_LOC 2
#define F1_LOC 3
#define N_LOC 2
#define M_LOC 3

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

        // TODO make following 3 functions code look better
        void splitConstantActivations(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, DenseElementsAttr at) {
            ShapedType initialShape = at.getType();
            ArrayRef<int64_t> s = initialShape.getShape();

            uint64_t COut = s[COUT_LOC];
            uint64_t CIn = s[CIN_LOC];
            uint64_t N = s[N_LOC];
            uint64_t M = s[M_LOC];

            uint64_t COut_switch = COut;
            uint64_t CIn_switch = CIn;
            uint64_t N_switch = N;
            uint64_t M_switch = M;

            if(split == PSplit) {
                COut_switch = COut / 2;
            } else {
                CIn_switch = CIn / 2;
            }

            if(initialShape.getElementType().isF32()) { // TODO is this the only choice?
                std::vector<APFloat> left;
                std::vector<APFloat> right;

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    //llvm::outs() << "Got this value: ";
                    //(*it).print(llvm::outs());

                    uint64_t loc_cout = i / (N * M * CIn);
                    uint64_t loc_cin = (i / (N * M)) % CIn;
                    if((loc_cout < COut_switch) && (loc_cin < CIn_switch)) {
                        left.push_back(*it);
                    } else {
                        right.push_back(*it);
                    }

                    i++;
                }

                assert(left.size() == right.size() == (at.getType().getNumElements() / 2));

                // now splitted the dense in two part, need to regenerate it
                unsigned int modifiedShapeAt;
                if(split == PSplit) {
                    modifiedShapeAt = 0;
                } else {
                    modifiedShapeAt = 1;
                }
                ShapedType ttype = halveShapeAt(initialShape, modifiedShapeAt);

                DenseElementsAttr attrLeft = DenseElementsAttr::get(ttype, left);
                DenseElementsAttr attrRight = DenseElementsAttr::get(ttype, right);

                // Construct the new constants
                Operation* cstl = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrLeft);
                Operation* cstr = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrRight);

                ops.push_back(cstl->getResult(0));
                ops.push_back(cstr->getResult(0));
            }
        }

        void splitConstantWeights(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, DenseElementsAttr at) {
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

            if(split == PSplit) {
                COut_switch = COut / 2;
            } else {
                CIn_switch = CIn / 2;
            }

            uint64_t i = 0;
            if(initialShape.getElementType().isF32()) { // TODO is this the only choice?
                std::vector<APFloat> left;
                std::vector<APFloat> right;

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    //llvm::outs() << "Got this value: ";
                    //(*it).print(llvm::outs());

                    uint64_t loc_cout = i / (F0 * F1 * CIn);
                    uint64_t loc_cin = (i / (F0 * F1)) % CIn;
                    if((loc_cout < COut_switch) && (loc_cin < CIn_switch)) {
                        left.push_back(*it);
                    } else {
                        right.push_back(*it);
                    }

                    i++;
                }

                assert(left.size() == right.size() == (at.getType().getNumElements() / 2));

                // now splitted the dense in two part, need to regenerate it
                unsigned int modifiedShapeAt;
                if(split == PSplit) {
                    modifiedShapeAt = 0;
                } else {
                    modifiedShapeAt = 1;
                }
                ShapedType ttype = halveShapeAt(initialShape, modifiedShapeAt);
                DenseElementsAttr attrLeft = DenseElementsAttr::get(ttype, left);
                DenseElementsAttr attrRight = DenseElementsAttr::get(ttype, right);

                // Construct the new constants
                Operation* cstl = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrLeft);
                Operation* cstr = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrRight);

                ops.push_back(cstl->getResult(0));
                ops.push_back(cstr->getResult(0));
            }

        }

        void splitConstantBias(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, DenseElementsAttr at) {
            ShapedType initialShape = at.getType();
            if(initialShape.getElementType().isF32()) { // TODO is this the only choice?
                std::vector<APFloat> left;
                std::vector<APFloat> right;

                uint64_t i = 0;
                for(auto it =  at.float_value_begin(); it != at.float_value_end(); it++) {
                    if(split == PSplit) {
                        if(i < (at.getType().getNumElements() / 2)) {
                            left.push_back(*it);
                        } else {
                            right.push_back(*it);
                        }
                    } else {
                        left.push_back(*it);
                        right.push_back(APFloat((float)0)); // NOTE assume that same kernel with 0 bias from the compiler point of view
                    }
                    i++;
                }

                assert((split == PSplit) && (left.size() == right.size() == (at.getType().getNumElements() / 2)) ||
                       (split == CaSplit) && (left.size() == right.size() == at.getType().getNumElements()));

                // now splitted the dense in two part, need to regenerate it
                ShapedType ttype;
                if(split == PSplit) {
                    ttype = halveShapeAt(initialShape, 0);
                } else {
                    ttype = initialShape;
                }

                DenseElementsAttr attrLeft = DenseElementsAttr::get(ttype, left);
                DenseElementsAttr attrRight = DenseElementsAttr::get(ttype, right);

                // Construct the new constants
                Operation* cstl = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrLeft);
                Operation* cstr = builder.create<ConstantOp>(builder.getUnknownLoc(), ttype, attrRight);

                ops.push_back(cstl->getResult(0));
                ops.push_back(cstr->getResult(0));
            }

        }

        void splitConstant(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, SplitType t) {
            for(NamedAttribute attr: op->getAttrs()) {
                // We Look for the Dense Attribute
                auto at = attr.second.dyn_cast<DenseElementsAttr>();
                if(at) {
                    if(t == bSplitType) {
                        splitConstantBias(op, ops, builder, split, at);
                    } else if(t == aSplitType) {
                        splitConstantActivations(op, ops, builder, split, at);
                    } else {
                        splitConstantWeights(op, ops, builder, split, at);
                    }

                }
            }
        }
    }
}

