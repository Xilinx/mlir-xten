// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

// This is an empty module pass.  Copy it to make your own pass.

#include "ATenDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include <vector>
#include <set>
#include <map>

#define DEBUG_TYPE "return-elimination"

using namespace mlir;

namespace {

class ReturnEliminationPass : public ModulePass<ReturnEliminationPass> {

public:
  ReturnEliminationPass() {}

  void runOnOperation(Operation *op) {

    if (visitedOps.count(op))
      return;
    visitedOps.insert(op);

    if (auto callOp = dyn_cast<CallOp>(op)) {

      auto builder = std::make_unique<mlir::OpBuilder>(op);

      std::vector<Type> tys;
      for (auto t : callOp.getCalleeType().getInputs())
        tys.push_back(t);
      for (auto t : callOp.getCalleeType().getResults())
        tys.push_back(t);

      auto newFnTy = FunctionType::get(tys, {}, op->getContext());
      std::string newFnName = callOp.callee().str()+"_out";

      if (!getModule().lookupSymbol<FuncOp>(newFnName)) {
        auto fn = FuncOp::create(op->getLoc(), newFnName, newFnTy);
        getModule().push_back(fn);
      }

      std::vector<Value> newCallArgs{callOp.arg_operand_begin(),
                                      callOp.arg_operand_end()};

      for (auto v : callOp.getResults()) {
        if (!v.getType().isa<MemRefType>())
          llvm_unreachable("function returns non-memref");
        if (!valueMap.count(v)) {
          valueMap[v] = builder->create<AllocOp>(op->getLoc(),
                                                 v.getType().cast<MemRefType>());
        }
        v.replaceAllUsesWith(valueMap[v]);
        newCallArgs.push_back(valueMap[v]);
      }

      auto newCallOp = builder->create<CallOp>(op->getLoc(),
                                               newFnName,
                                               ArrayRef<Type>{},
                                               newCallArgs);
      erasedOps.insert(op);
      auto fn = getModule().lookupSymbol<FuncOp>(callOp.callee());
      if (fn && fn.use_empty()) erasedOps.insert(fn);
    }
    else if ( isa<AllocOp>(op) ) {
      Value v = op->getResult(0);
      if (valueMap.count(v)) {
        v.replaceAllUsesWith(valueMap[v]);
        erasedOps.insert(op);
      }
    }
    else if ( isa<xilinx::aten::AcapAllocOp>(op) ) {
    }
    else {
      //getModule().dump();
      //op->dump();
      //llvm_unreachable("unhandled operation type");
    }

    for (Value v : op->getOperands()) {
      if (!v.getType().isa<MemRefType>())
        continue;
      if (v.isa<BlockArgument>())
        continue;
      runOnOperation(v.getDefiningOp());
    }

  }

  void runOnModule() override {

    auto module = getModule();
    auto context = module.getContext();

    // check that a function called "graph" exists
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    // assume a single bb with a single return statement
    Block &BB = graph.front();

    FunctionType funcTy = graph.getType();
    std::vector<Type> newFuncInputTys;

    for (auto ty : funcTy.getInputs())
      newFuncInputTys.push_back(ty);

    for (auto ty : funcTy.getResults())
      newFuncInputTys.push_back(ty);

    FunctionType newFuncTy = FunctionType::get(newFuncInputTys, {}, module.getContext());
    graph.setType(newFuncTy);

    Operation *retOp = BB.getTerminator();
    auto builder = std::make_unique<mlir::OpBuilder>(retOp);

    builder->create<ReturnOp>(retOp->getLoc());

    std::vector<Value> operands{retOp->getOperands().begin(),
                                 retOp->getOperands().end()};

    retOp->dropAllReferences();
    erasedOps.insert(retOp);

    for (Value v : operands)
      valueMap[v] = BB.addArgument(v.getType());


    for (Value v : operands) {
      if (!v.getType().isa<MemRefType>())
        llvm_unreachable("graph function returns non-memref");
      runOnOperation(v.getDefiningOp());
    }

    for (auto oi=BB.rbegin(),oe=BB.rend(); oi!=oe; ++oi) {
      Operation *o = &*oi;
      for (Value v : o->getResults()) {
        if (v.getType().isa<MemRefType>()) {
          runOnOperation(o);
          break;
        }
      }
    }

    for (Operation *o : erasedOps)
      o->erase();
  }

private:
  std::map<Value,Value> valueMap;
  std::set<Operation*> visitedOps;
  std::set<Operation*> erasedOps;
};

} // namespace

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createReturnEliminationPass() {
  return std::make_unique<ReturnEliminationPass>();
}

} // namespace aten
} // namespace xilinx
