// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

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
#include <torch-mlir/Dialect/Torch/Utils/Utils.h>
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

Value toTorchTensorTypeCast(PatternRewriter &rewriter, Value input) {

  auto tensorTy = dyn_cast<ShapedType>(input.getType());
  auto sizes = tensorTy.getShape();

  return rewriter
      .create<TorchConversion::FromBuiltinTensorOp>(
          input.getLoc(),
          mlir::torch::Torch::ValueTensorType::get(input.getContext(), sizes,
                                                   tensorTy.getElementType()),
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

struct Conv2dPadding {
  std::array<int64_t, 2> hPadding;
  std::array<int64_t, 2> wPadding;

  [[nodiscard]] bool isSymmetric() const {
    return hPadding[0] == hPadding[1] && wPadding[0] == wPadding[1];
  }
};

Conv2dPadding getPadding(GroupConv2dOp::Adaptor adaptor) {
  auto pad = adaptor.getPad();
  assert(pad.size() == 2 && "expected 2 elements by definition");

  auto hPadding = cast<DenseI64ArrayAttr>(pad[0]);
  auto wPadding = cast<DenseI64ArrayAttr>(pad[1]);
  assert(hPadding.size() == 2 && "expected 2 elements by definition");
  assert(wPadding.size() == 2 && "expected 2 elements by definition");

  return {{hPadding[0], hPadding[1]}, {wPadding[0], wPadding[1]}};
}

template <typename SrcOpT>
ValueRange oneToOneXTenNNToTorch(SrcOpT op,
                                 typename SrcOpT::Adaptor /*adaptor*/,
                                 ArrayRef<Type> types, ValueRange values,
                                 ConversionPatternRewriter &rewriter) {
  // Start composing new op
  OperationState state(
      op->getLoc(), "torch.aten." + std::string(op->getName().stripDialect()),
      values, types, op->getAttrs(), op->getSuccessors());

  // Create the new op
  return rewriter.create(state)->getResults();
}

ValueRange groupConv2dToTorch(GroupConv2dOp op, GroupConv2dOp::Adaptor adaptor,
                              ArrayRef<Type> types, ValueRange values,
                              ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();

  auto newInput = values[0];
  mlir::Value conv2dPads;
  Conv2dPadding structPadding = getPadding(adaptor);
  if (!structPadding.isSymmetric()) {
    // Padding is not symmetric which is the only mode aten conv2d op supports.
    // We circumvent this problem by adding a padding operation

    // Build new vtensor result type
    auto ty = cast<Torch::ValueTensorType>(newInput.getType());
    mlir::Type paddingResultTy;
    std::optional<llvm::ArrayRef<int64_t>> optSizes = ty.getOptionalSizes();
    if (optSizes) {
      auto newSizes = ty.getSizes().vec();
      newSizes[2] += structPadding.hPadding[0] + structPadding.hPadding[1];
      newSizes[3] += structPadding.wPadding[0] + structPadding.wPadding[1];
      paddingResultTy = Torch::ValueTensorType::get(op->getContext(), newSizes,
                                                    ty.getOptionalDtype());
    } else {
      paddingResultTy = Torch::ValueTensorType::get(
          op->getContext(), ty.getOptionalSizes(), ty.getOptionalDtype());
    }

    auto zeroPadValue = rewriter.create<Torch::ConstantIntOp>(loc, 0);
    auto pads = Torch::toTorchList(
        loc, rewriter,
        {structPadding.hPadding[0], structPadding.hPadding[1],
         structPadding.wPadding[0], structPadding.wPadding[1]});
    newInput = rewriter.create<Torch::AtenConstantPadNdOp>(
        loc, paddingResultTy, newInput, pads, zeroPadValue);

    // We want zero pad for the Conv2d since we are going to apply it with a
    // padding op
    conv2dPads = Torch::toTorchList(loc, rewriter, {0, 0});
  } else {
    conv2dPads = Torch::toTorchList(
        loc, rewriter, {structPadding.hPadding[0], structPadding.wPadding[0]});
  }

  auto newWeights = values[1];
  auto newBias = values[2];
  auto stride = Torch::toTorchList(loc, rewriter, adaptor.getStride().vec());
  auto dilation =
      Torch::toTorchList(loc, rewriter, adaptor.getDilation().vec());
  auto group =
      rewriter.create<Torch::ConstantIntOp>(loc, adaptor.getGroupAttr());

  return rewriter
      .create<Torch::AtenConv2dOp>(loc, types[0], newInput, newWeights, newBias,
                                   stride, conv2dPads, dilation, group)
      ->getResults();
}

template <typename SrcOpT, ValueRange codegenFunc(
                               SrcOpT, typename SrcOpT::Adaptor, ArrayRef<Type>,
                               ValueRange, ConversionPatternRewriter &)>
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
    llvm::transform(op->getResultTypes(),
                    std::back_inserter(vtensorResultTypes), [&](Type ty) {
                      auto tensorTy = cast<TensorType>(ty);
                      return Torch::ValueTensorType::get(
                          ctx, tensorTy.getShape(), tensorTy.getElementType());
                    });

    // Call the function that creates the new operation.
    auto newValues =
        codegenFunc(op, adaptor, vtensorResultTypes, vtensorOperands, rewriter);

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

    patterns.add<ApplyXTenNNToTorch<GroupConv2dOp, groupConv2dToTorch>>(
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
