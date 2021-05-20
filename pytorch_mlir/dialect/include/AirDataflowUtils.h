#ifndef AIR_DATAFLOW_UTILS
#define AIR_DATAFLOW_UTILS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "AIRDialect.h"
#include "AirOpWrapper.h"
#include "AirDataflowConsts.h"

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
            // unsigned int K; // Probably enough as an implicit parameter
            // unsigned int lineGrannularity; // TODO add this?
            // TODO is false then should disabel W optimisation? Or should we find something more detailed for it?

            ModelParams() {
                P = 1;
                Ca = 1;
                W = 1;
                L = 1;
            }

            ModelParams(unsigned int defP, unsigned int defCa, unsigned int defL, unsigned int defW) {
                this->P = defP;
                this->Ca = defCa;
                this->L = defL;
                this->W = defW;
            }

            unsigned int cores() {
                return P * Ca * L * W;
            }

            bool nonZero() {
                return (P != 0) && (Ca != 0) && (L != 0) && (W != 0);
            }

            void print() {
                llvm::outs() << "P: " << this->P << ", Ca: " << this->Ca << ", L: " << this->L << ", W: " << this->W << "\n";
            }
        };

        ShapedType breakShapeInto(ShapedType initShape, unsigned int at, unsigned int into);
        ShapedType mergeShapeInto(ShapedType initShape, unsigned int at, unsigned int into);

        void splitConstantInto(ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, SplitType t, unsigned int into);

        void deleteOpsFrom(std::vector<Operation*> &ops);
        void deleteOpsFrom(std::vector<AbsOpWrapper*> &ops);

        void insertConcat(OpBuilder &builder, Value prevRes, std::vector<Value> &values, unsigned int dim);
        void replaceConcat(OpBuilder &builder, xilinx::air::ConcatOp concat, std::vector<Value> &nInputs,
                           std::vector<Operation*> &toDelete, unsigned int dim, unsigned int into);

        void insertSplit(OpBuilder &builder, Value prevInput, std::vector<Value> &nInputs, unsigned int dim, unsigned int into);
        void replaceSplit(OpBuilder &builder, xilinx::air::SplitOp split, std::vector<Value> &values,
                          std::vector<Operation*> &toDelete, unsigned int dim);
    }
}

#endif
