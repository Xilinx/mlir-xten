#ifndef AIR_DATAFLOW_UTILS
#define AIR_DATAFLOW_UTILS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "AIRDialect.h"
#include "AirOpWrapper.h"

// Weight locations
#define COUT_LOC 0
#define CIN_LOC 1
#define F0_LOC 2
#define F1_LOC 3

// Acts locs
#define BATCH_LOC 0
#define C_LOC 1
#define N_LOC 2
#define M_LOC 3

using namespace mlir;

namespace xilinx {
    namespace air {

        enum Split {PSplit, CaSplit, LSplit, WSplit};
        enum SplitType {wSplitType, aSplitType, bSplitType};

        class ModelParams {
        public:
            unsigned int P;
            unsigned int Ca;
            unsigned int L;
            unsigned int W;
            // unsigned int K; // Probably enough as an with implicit parameter

            ModelParams() {
                P = 1;
                Ca = 1;
                W = 1;
                L = 1;
            }
        };

        ShapedType breakShapeInto(ShapedType initShape, unsigned int at, unsigned int into);
        ShapedType mergeShapeInto(ShapedType initShape, unsigned int at, unsigned int into);

        void splitConstantInto(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, SplitType t, unsigned int into);

        void deleteOpsFrom(std::vector<Operation*> &ops);
        void deleteOpsFrom(std::vector<AbsOpWrapper*> &ops);

        void insertConcat(OpBuilder &builder, Value prevRes, std::vector<Value> &values, unsigned int dim);
        void replaceConcat(OpBuilder &builder, xilinx::air::ConcatOp concat, std::vector<Value> nInputs,
                           std::vector<Operation*> toDelete, unsigned int dim, unsigned int into);

        void insertSplit(OpBuilder &builder, Value prevInput, std::vector<Value> &nInputs, unsigned int dim, unsigned int into);
        void replaceSplit(OpBuilder &builder, xilinx::air::SplitOp split, std::vector<Value> &values,
                          std::vector<Operation*> &toDelete, unsigned int dim);
    }
}

#endif
