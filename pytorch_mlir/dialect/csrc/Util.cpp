#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-util"

using namespace mlir;

namespace xilinx {
namespace air {

void coalesceLoops(AffineForOp outer, AffineForOp inner)
{
  auto ctx = outer.getContext();
  auto loc = outer.getLoc();
  auto builder = OpBuilder::atBlockBegin(outer.getBody());
  // ub_new = ub_inner*ub_outer
  // iv_new = 0...ub_new-1
  // iv_new_inner = mod(iv_new, ub_inner)
  // iv_new_outer = floordiv(iv_new, ub_inner)
  auto ub_inner_expr = inner.getUpperBoundMap().getResult(0);
  auto ub_outer_expr = outer.getUpperBoundMap().getResult(0);
  auto ub_new_expr = ub_inner_expr * ub_outer_expr;
  auto iv_new_inner_expr = getAffineDimExpr(0, ctx) % ub_inner_expr;
  auto iv_new_outer_expr = getAffineDimExpr(0, ctx).floorDiv(ub_inner_expr);

  outer.setUpperBoundMap(AffineMap::get(0, 0, ub_new_expr));
  auto iv_new = outer.getInductionVar();
  auto iv_new_inner = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, iv_new_inner_expr),
                                                    iv_new);
  auto iv_new_outer = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, iv_new_outer_expr),
                                                    iv_new);
  SmallPtrSet<Operation *, 2> keep{iv_new_inner,iv_new_outer};
  iv_new.replaceAllUsesExcept(iv_new_outer, keep);
  inner.getInductionVar().replaceAllUsesWith(iv_new_inner);
  // erase terminator from inner loop's body
  inner.getBody()->back().erase();
  // move inner loop's body to outer loop
  outer.getBody()->getOperations().splice(Block::iterator(inner.getOperation()),
                                          inner.getBody()->getOperations());
  inner.erase();
  return;
}

void normalizeLoop(AffineForOp afo)
{
  auto ubMap = afo.getUpperBoundMap();
  auto lbMap = afo.getLowerBoundMap();
  auto ctx = afo.getContext();
  auto loc = afo.getLoc();

  auto step_expr = getAffineConstantExpr(afo.getStep(), ctx);

  auto ub_expr = ubMap.getResult(0);
  auto lb_expr = lbMap.getResult(0);
  auto sub_expr = ub_expr - lb_expr;
  auto new_ub_expr = sub_expr.ceilDiv(step_expr);

  auto iv = afo.getInductionVar();

  afo.setLowerBoundMap(AffineMap::get(0, 0, getAffineConstantExpr(0, ctx)));
  afo.setUpperBoundMap(AffineMap::get(0, 0, new_ub_expr));
  afo.setStep(1);

  auto dim0_expr = getAffineDimExpr(0, ctx);
  auto iv_expr = dim0_expr * step_expr + lb_expr;
  auto iv_map = AffineMap::get(1, 0, iv_expr);
  auto builder = OpBuilder::atBlockBegin(afo.getBody());
  auto new_iv = builder.create<AffineApplyOp>(loc, iv_map, iv);
  SmallPtrSet<Operation *, 1> keep{new_iv};
  iv.replaceAllUsesExcept(new_iv, keep);
  return;
}

}
}