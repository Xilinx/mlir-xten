#include "AirDataflowUtils.h"
#include "mlir/IR/OperationSupport.h"


#define DEBUG_TYPE "air-dataflow-utils"

using namespace mlir;

namespace xilinx {
    namespace air {
        ShapedType halveShapeAt(ShapedType initShape, unsigned int at) {
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

        void splitConstant(ConstantOp op, std::vector<Operation*> &ops, OpBuilder &builder) {
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
                        ShapedType ttype = halveShapeAt(initialShape, 0);

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
    }
}

