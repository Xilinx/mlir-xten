#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace xilinx {
namespace air {

void coalesceLoops(mlir::AffineForOp outer, mlir::AffineForOp inner);

void normalizeLoop(AffineForOp afo);

}
}