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

template <typename SrcOpT>
ValueRange oneToOneXTenNNToTorch(SrcOpT op,
                                 typename SrcOpT::Adaptor /*adaptor*/,
                                 ArrayRef<Type> types, ValueRange values,
                                 OpBuilder *rewriter) {
  // Start composing new op
  OperationState state(
      op->getLoc(), "torch.aten." + std::string(op->getName().stripDialect()),
      values, types, op->getAttrs(), op->getSuccessors());

  // Create the new op
  return rewriter->create(state)->getResults();
}

template <typename SrcOpT,
          ValueRange codegenFunc(SrcOpT, typename SrcOpT::Adaptor,
                                 ArrayRef<Type>, ValueRange, OpBuilder *)>
class ApplyXTenNNToTorch : public OpConversionPattern<SrcOpT> {
public:
  using OpConversionPattern<SrcOpT>::OpConversionPattern;
  using OpAdaptor = typename SrcOpT::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto *ctx = op->getContext();

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

    // Call the function that creates the new operation.
    auto newValues = codegenFunc(op, adaptor, vtensorResultTypes,
                                 vtensorOperands, &rewriter);

    // Convert Torch builtin types back to MLIR types retrieving the
    // original type of the op.
    SmallVector<Value> vtensorResults;
    llvm::transform(llvm::enumerate(newValues),
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
#define INSERT_ONE_TO_ONE_PATTERN(XTenOp)                                      \
  target.addIllegalOp<XTenOp>();                                               \
  patterns.add<ApplyXTenNNToTorch<                                             \
      XTenOp, oneToOneXTenNNToTorch<amd::xten_nn::XTenOp>>>(context);
    INSERT_ONE_TO_ONE_PATTERN(Atan2Op)
    INSERT_ONE_TO_ONE_PATTERN(CosOp)
    INSERT_ONE_TO_ONE_PATTERN(MishOp)
    INSERT_ONE_TO_ONE_PATTERN(RoundOp)
    INSERT_ONE_TO_ONE_PATTERN(SignOp)
    INSERT_ONE_TO_ONE_PATTERN(SinOp)
#undef INSERT_UNARY_PATTERN

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
