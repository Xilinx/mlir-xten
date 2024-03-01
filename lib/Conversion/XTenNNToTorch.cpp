#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>
#include <torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h>

#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Util/Util.h"

namespace xilinx::xten {
#define GEN_PASS_DECL_CONVERTXTENNNTOTORCH
#define GEN_PASS_DEF_CONVERTXTENNNTOTORCH
#include "xten/Conversion/Passes.h.inc"
} // namespace xilinx::xten

using namespace mlir;
using namespace amd::xten_nn;
using namespace mlir::torch;

namespace {

Type getCompatibleTorchDType(MLIRContext *ctx, Type dtype) {
  if (isa<FloatType>(dtype))
    return dtype;

  auto integerType = dtype.cast<IntegerType>();
  if (!integerType.isSignless())
    return dtype;

  // Torch builtin types only support Signed and Unsigned integer types.
  // Convert Signless to Signed.
  return IntegerType::get(ctx, integerType.getWidth(),
                          mlir::IntegerType::Signed);
}

Value toTorchTensorTypeCast(PatternRewriter &rewriter, Value input) {

  auto tensorTy = dyn_cast<ShapedType>(input.getType());
  auto sizes = tensorTy.getShape();

  auto dtype =
      getCompatibleTorchDType(rewriter.getContext(), tensorTy.getElementType());

  return rewriter
      .create<TorchConversion::FromBuiltinTensorOp>(
          input.getLoc(),
          mlir::torch::Torch::ValueTensorType::get(input.getContext(), sizes,
                                                   dtype),
          input)
      .getResult();
}

Value toBuiltinTensorTypeCast(OpBuilder &builder, Value val, Type type) {
  if (val.getType().isa<MemRefType>())
    return val;

  auto tensorTy = val.getType().dyn_cast<torch::Torch::BaseTensorType>();
  if (!tensorTy)
    return val;
  return builder.create<torch::TorchConversion::ToBuiltinTensorOp>(val.getLoc(),
                                                                   type, val);
}

template <typename... ops>
class GenericPatternXTenNNToTorch : public RewritePattern {
public:
  GenericPatternXTenNNToTorch(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto *ctx = op->getContext();
    if (isa<ops...>(op)) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "no conversion for this operation.");
    }

    if (op->getName().getDialectNamespace() != "xten_nn") {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "operation doesn't belong to XTenNN dialect.");
    }

    SmallVector<Value> vtensorOperands;
    llvm::transform(
        op->getOperands(), std::back_inserter(vtensorOperands),
        [&](Value val) { return toTorchTensorTypeCast(rewriter, val); });

    // Convert MLIR types to Torch builtin types.
    SmallVector<Type> vtensorResultTypes;
    llvm::transform(
        op->getResultTypes(), std::back_inserter(vtensorResultTypes),
        [&](Type ty) {
          auto tensorTy = cast<TensorType>(ty);
          return Torch::ValueTensorType::get(
              ctx, tensorTy.getShape(),
              getCompatibleTorchDType(ctx, tensorTy.getElementType()));
        });

    // Start composing new op
    OperationState state(
        op->getLoc(), "torch.aten." + std::string(op->getName().stripDialect()),
        vtensorOperands, vtensorResultTypes, op->getAttrs(),
        op->getSuccessors());

    // Create the new op
    Operation *newOp = rewriter.create(state);

    // Convert Torch builtin types back to MLIR types retrieving the original type of the op.
    SmallVector<Value> vtensorResults;
    llvm::transform(llvm::enumerate(newOp->getResults()),
                    std::back_inserter(vtensorResults), [&](const auto it) {
                      return toBuiltinTensorTypeCast(
                          rewriter, it.value(),
                          op->getResult(it.index()).getType());
                    });
    rewriter.replaceOp(op, vtensorResults);
    return success();
  }
};

struct ConvertXTenNNToTorch
    : public xilinx::xten::impl::ConvertXTenNNToTorchBase<
          ConvertXTenNNToTorch> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amd::xten_nn::XTenNNDialect, torch::Torch::TorchDialect,
                    tensor::TensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    ConversionTarget target(*context);
    target.addLegalOp<SubgraphOp>();
    target.addLegalDialect<Torch::TorchDialect,
                           TorchConversion::TorchConversionDialect,
                           func::FuncDialect>();

    RewritePatternSet patterns(context);
    patterns.add<GenericPatternXTenNNToTorch<SubgraphOp, LoadExternalConstOp>>(
        context);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace amd {
namespace xten_nn {

std::unique_ptr<mlir::Pass> createXTenNNToTorchPass() {
  return std::make_unique<ConvertXTenNNToTorch>();
}

} // namespace xten_nn
} // namespace amd
