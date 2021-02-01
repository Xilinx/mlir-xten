#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace xilinx {
namespace air {

void coalesceLoops(mlir::AffineForOp outer, mlir::AffineForOp inner);

void normalizeLoop(AffineForOp afo);

FuncOp getATenFn(ModuleOp module, std::string fnName, ArrayRef<Value> operands, ArrayRef<Type> retTys);

}
}