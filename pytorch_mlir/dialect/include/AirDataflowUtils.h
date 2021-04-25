#ifndef AIR_DATAFLOW_UTILS
#define AIR_DATAFLOW_UTILS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace xilinx {
    namespace air {
        enum Split {PSplit, CaSplit};
        enum SplitType {wSplitType, aSplitType, bSplitType};

        ShapedType halveShapeAt(ShapedType initShape, unsigned int at);
        void splitConstant(ConstantOp op, std::vector<Operation*> &ops, OpBuilder &builder, Split split, SplitType t);
    }
}

#endif
