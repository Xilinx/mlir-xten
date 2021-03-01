// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"
#include "AffineToAIRPass.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "affine-to-air"

namespace {

#include "AffineToAIR.cpp.inc"

class AffineCopyToAIRDMAConversion : public ConversionPattern {
public:
  explicit AffineCopyToAIRDMAConversion(MLIRContext *context)
      : ConversionPattern(AffineDmaStartOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto affine_dma_start = cast<AffineDmaStartOp>(op);

    auto src = affine_dma_start.getSrcMemRef();
    auto src_indices = affine_dma_start.getSrcIndices();

    auto dst = affine_dma_start.getDstMemRef();
    auto dst_indices = affine_dma_start.getDstIndices();

    if (!affine_dma_start.getSrcMap().isIdentity())
      return failure();

    if (src_indices.size() != 1 ||
        dst_indices.size() != 1)
      return failure();

    SmallVector<Type,1> tys;
    SmallVector<Value,1> deps;
    rewriter.create<air::DmaMemcpyOp>(op->getLoc(), tys,
                                       deps, dst, src, 
                                       dst_indices.front(),
                                       src_indices.front(),
                                       affine_dma_start.getNumElements());
    rewriter.eraseOp(op);
    return success();
  }
    // mlir::AffineLoadOp load = nullptr;
    // mlir::AffineStoreOp store = nullptr;

    // for (auto &o : afo.getLoopBody().getOps()) {
    //   if (isa<AffineLoadOp>(o)) {
    //     load = cast<AffineLoadOp>(o);
    //   }
    //   else if (isa<AffineStoreOp>(o)) {
    //     store = cast<AffineStoreOp>(o);
    //   }
    //   else if (isa<AffineYieldOp>(o)) {
    //   }
    //   else {
    //     llvm::outs() << "FAIL!\n";
    //     op->print(llvm::outs());
    //     o.print(llvm::outs());
    //     return failure();
    //   }
    // }

    // llvm::outs() << "HERE!\n";
    // op->print(llvm::outs());

    // if (!load || !store)
    //   return failure();

    // if (store.value() != load)
    //   return failure();

    // auto srcTy = load.memref().getType().cast<mlir::MemRefType>();
    // auto dstTy = store.memref().getType().cast<mlir::MemRefType>();

    // if (srcTy.getMemorySpace() == 0 && dstTy.getMemorySpace() == 1) {
    //   // ext -> L2
    //   // #map7 = affine_map<()[s0] -> (s0 + 32)>
    //   // affine.for %arg5 = %arg3 to #map7()[%arg3] {
    //   //   %0 = affine.load %arg1[%arg4, %arg5] : memref<256x256xf32>
    //   //   affine.store %0, %arg2[-%arg0 + %arg4, -%arg3 + %arg5] : memref<32x32xf32, 1>
    //   // }
    //   //air.shim_dma_memcpy(%src,  %dst,  %src_d1, %src_d0, %dst_d1,        %dst_d0, %num)
    //   //air.shim_dma_memcpy(%arg1, %arg2, %arg4,   %arg3,   -%arg0 + %arg4, 0,       32)
    //   llvm::outs() << "L3 to L2!\n";

    //   mlir::AffineMap lbm = afo.getLowerBoundMap();
    //   mlir::AffineMap ubm = afo.getUpperBoundMap();

    //   auto int32Ty = mlir::IntegerType::get(op->getContext(), 32);
    //   auto attr = mlir::IntegerAttr::get(int32Ty, 0);
    //   SmallVector<Attribute, 1> attrs{attr};
    //   SmallVector<Attribute, 2> ints;
    //   lbm.constantFold(attrs, ints);
    //   ubm.constantFold(attrs, ints);
    //   int64_t lower_bound = ints[0].cast<mlir::IntegerAttr>().getInt();
    //   int64_t upper_bound = ints[1].cast<mlir::IntegerAttr>().getInt();

    //   llvm::outs() << "LB: " << lower_bound << " UB: " << upper_bound << "\n";
    //   auto loc = op->getLoc();
    //   auto zero_const = rewriter.create<ConstantIndexOp>(loc, 0);
    //   auto upper_bound_const = rewriter.create<ConstantIndexOp>(loc, upper_bound);
    //   SmallVector<Value, 1> deps;
    //   SmallVector<Type, 1> rets;
    //   /*auto shim_dma_memcpy =*/ rewriter.create<xilinx::air::DmaMemcpy2d>(loc, rets, deps, load.memref(), store.memref(),
    //                                                                    load.indices()[0], afo.getLowerBoundOperands()[0],
    //                                                                    store.indices()[0], zero_const, upper_bound_const);
    //   // rewriter.eraseOp(load);
    //   // rewriter.eraseOp(store);
    //   rewriter.eraseOp(op);

    //   return success();
    // }
    // else if (srcTy.getMemorySpace() == 1 || dstTy.getMemorySpace() == 0) {
    //   // L2 -> ext
    // }
    // else {
    //   return failure();
    // }
    // return failure();
  //}
};

class AffineParToHerdLaunchConversion : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDims() == 2) {
      auto loc = op.getLoc();
      auto ub0 = op.upperBoundsMap().getResult(0).cast<AffineConstantExpr>();
      auto ub1 = op.upperBoundsMap().getResult(1).cast<AffineConstantExpr>();
      SmallVector<Value, 4> args;
      SmallVector<Value, 4> constants;
      llvm::SetVector<Value> region_args;
      getUsedValuesDefinedAbove(op.getRegion(), region_args);
      for (Value v : region_args) {
        if (v.getDefiningOp() && isa<ConstantOp>(v.getDefiningOp()))
          constants.push_back(v);
        else
          args.push_back(v);
      }
      air::HerdDim2 dims{rewriter.create<ConstantIndexOp>(loc,ub0.getValue()),
                         rewriter.create<ConstantIndexOp>(loc,ub1.getValue())};
      auto launch = rewriter.create<air::HerdLaunchOp>(op.getLoc(), dims, args);
      auto &bb = launch.body().front();
      auto ivs = op.getIVs();
      ivs[0].replaceAllUsesWith(launch.getTileIds().x);
      ivs[1].replaceAllUsesWith(launch.getTileIds().y);
      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body,
                                body.begin(), --body.end());
      rewriter.setInsertionPointToStart(&launch.getRegion().front());
      for (auto c : constants) {
        replaceAllUsesInRegionWith(c,
                                   rewriter.clone(*c.getDefiningOp())->getResult(0),
                                   launch.getRegion());
      }
      auto builder = OpBuilder::atBlockEnd(&bb);
      builder.create<air::HerdTerminatorOp>(loc);

      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

      rewriter.eraseOp(op);
    
      return success();
    }
    return failure();
  }
};

struct AffineToAIRPass : public PassWrapper<AffineToAIRPass,
                                            OperationPass<ModuleOp>> {


  LogicalResult lower_dma_to_function(StringRef callee, CallOp dma_callOp)
  {
    auto module = getOperation();
    auto funcOp = module.lookupSymbol<mlir::FuncOp>(callee);
    auto ctx = funcOp.getContext();
    auto loc = dma_callOp.getLoc();

    assert(callee.startswith("air_dma_copy"));
    for (auto &bb : funcOp) {
      for (auto &op : bb) {
        if (auto forOp = dyn_cast<AffineForOp>(op)) {
          mlir::AffineLoadOp load = nullptr;
          mlir::AffineStoreOp store = nullptr;

          for (auto &o : forOp.getLoopBody().getOps()) {
            if (isa<AffineLoadOp>(o)) {
              load = cast<AffineLoadOp>(o);
            }
            else if (isa<AffineStoreOp>(o)) {
              store = cast<AffineStoreOp>(o);
            }
            else if (isa<AffineYieldOp>(o)) {
            }
            else {
              return failure();
            }
          }


          if (!load || !store)
            return failure();

          if (store.value() != load)
            return failure();

          auto srcTy = load.memref().getType().cast<mlir::MemRefType>();
          auto dstTy = store.memref().getType().cast<mlir::MemRefType>();
          forOp->print(llvm::outs());
        }
      }
    }

    // for now it's all very much hard coded
    if ( callee.equals("acap_L2_dma_copy_1") ) {
      auto arg_iter = dma_callOp.arg_operand_begin();
      // input and output here are relative to the copy
      auto dim1_idx = *(arg_iter);
      auto input_operand = *(++arg_iter);
      auto output_operand = *(++arg_iter);
      auto dim0_idx = *(++arg_iter);
      std::string dmafn_name = "acap_L2_dma_copy_arg0";
      FuncOp dmafn = module.lookupSymbol<FuncOp>(dmafn_name);
      if (!dmafn) {
        SmallVector<Type, 4> tys{input_operand.getType(),
                                 output_operand.getType(),
                                 dim1_idx.getType(),
                                 dim0_idx.getType()};
        SmallVector<Type, 1> retTy{};
        auto fnTy = FunctionType::get(ctx, tys, retTy);
        dmafn = FuncOp::create(loc, dmafn_name, fnTy);
        dmafn.setPrivate();
        module.push_back(dmafn);
      } 
      OpBuilder builder(dma_callOp);
      SmallVector<Value,4> opers{input_operand, output_operand, dim1_idx, dim0_idx};
      SmallVector<Type, 1> retTy;
      builder.create<CallOp>(loc, retTy, builder.getSymbolRefAttr(dmafn_name), opers);
      dma_callOp.erase();
      //acap_L2_dma_copy_arg1(&weights);
    }
    else if (callee.equals("acap_L2_dma_copy")) {
      auto arg_iter = dma_callOp.arg_operand_begin();
      // input and output here are relative to the copy
      auto dim1_idx = *(arg_iter);
      auto input_operand = *(++arg_iter);
      auto dim0_idx = *(++arg_iter);
      auto output_operand = *(++arg_iter);
      std::string dmafn_name = "acap_L2_dma_copy_arg1";
      FuncOp dmafn = module.lookupSymbol<FuncOp>(dmafn_name);
      if (!dmafn) {
        SmallVector<Type, 4> tys{input_operand.getType(),
                                 output_operand.getType(),
                                 dim1_idx.getType(),
                                 dim0_idx.getType()};
        SmallVector<Type, 1> retTy{};
        auto fnTy = FunctionType::get(ctx, tys, retTy);
        dmafn = FuncOp::create(loc, dmafn_name, fnTy);
        dmafn.setPrivate();
        module.push_back(dmafn);
      } 
      OpBuilder builder(dma_callOp);
      SmallVector<Value,4> opers{input_operand, output_operand, dim1_idx, dim0_idx};
      SmallVector<Type, 1> retTy;
      builder.create<CallOp>(loc, retTy, builder.getSymbolRefAttr(dmafn_name), opers);
      dma_callOp.erase();
    }
  }

  LogicalResult lowerDma(StringRef callee, CallOp dma_callOp) {
    //return lowerDma_pad(callee, dma_callOp);
    return lower_dma_to_function(callee, dma_callOp);
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
     registry.insert<xilinx::air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    // // check that a function called "graph" exists
    // auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    // if (!graph) {
    //   emitError(mlir::UnknownLoc::get(context),
    //             "OpReportPass failed: can't find a graph function\n");
    //   signalPassFailure();
    //   return;
    // }

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto co = dyn_cast<CallOp>(op)) {
          if (co.getCallee().startswith("air_dma_copy")) {
            lowerDma(co.getCallee(), co);
          }
        }
      });
    }

    // tablegen patterns
    OwningRewritePatternList patterns;
    patterns.insert<AffineParToHerdLaunchConversion,
                    AffineCopyToAIRDMAConversion>(context);

    populateWithGenerated(context, patterns);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect,
                           StandardOpsDialect,
                           scf::SCFDialect>();

    target.addLegalOp<xilinx::air::DmaMemcpyOp>();
    target.addLegalOp<xilinx::air::DmaMemcpy2d>();
    target.addLegalOp<xilinx::air::HerdLaunchOp>();

    target.addLegalOp<AffineApplyOp,
                      AffineForOp,
                      AffineLoadOp,
                      AffineStoreOp,
                      AffineYieldOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    std::vector<AffineDmaWaitOp> waits;
    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto wo = dyn_cast<AffineDmaWaitOp>(op)) {
          waits.push_back(wo);
        }
      });
    }
    for (auto o : waits) o->erase();

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));

  }
};

}// namespace


namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineToAIRPass() {
  return std::make_unique<AffineToAIRPass>();
}

} // namespace air
} // namespace xilinx

void xilinx::air::registerAffineToAIRPass() {
    PassRegistration<AffineToAIRPass>(
      "affine-to-air",
      "Lift affine loops to AIR dialect");
}
