// (c) Copyright 2023 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

namespace xilinx::xten {
#define GEN_PASS_DECL_CONVERTXTENNNTOLINALG
#define GEN_PASS_DEF_CONVERTXTENNNTOLINALG
#include "xten/Conversion/Passes.h.inc"
} // namespace xilinx::xten

using namespace mlir;
using namespace amd::xten_nn;

namespace {

int64_t getRank(Value v) {
  return cast<ShapedType>(v.getType()).getRank();
}

int64_t getMaxRank(ValueRange operands) {
  int64_t maxRank = 0;
  for (Value operand : operands) {
    maxRank = std::max(maxRank, getRank(operand));
  }
  return maxRank;
}

bool isScalar(Value v) {
  return getRank(v) == 0;
}

Value getEmptyTensor(OpBuilder &b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                   dynSizes,
                                   type.cast<RankedTensorType>().getEncoding());
}

// Elu(x) = x > 0 ? x : alpha * (exp(x) - 1)
Value mapEluOpToArithAndMathOps(EluOp op, ArrayRef<Type> /*resultTypes*/,
                                Value operand, OpBuilder *b) {
  Location loc = op->getLoc();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (!isa<FloatType>(elementType)) {
    return nullptr;
  }

  // Build: exp(x) - 1
  Value exp = b->create<::mlir::math::ExpOp>(loc, operand);
  Value one =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 1));
  Value sub = b->create<::mlir::arith::SubFOp>(loc, exp, one);
  Value alphaAsValue = b->create<mlir::arith::ConstantFloatOp>(
      loc, EluOpAdaptor(op).getAlpha(), cast<FloatType>(elementType));
  Value mul = b->create<::mlir::arith::MulFOp>(loc, alphaAsValue, sub);

  // Build: x > 0 ? x : alpha * (exp(x) - 1)
  Value zero =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 0));
  Value cmpOp =
      b->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, operand, zero);
  return b->create<arith::SelectOp>(loc, cmpOp, operand, mul);
}

template <typename SrcOpT,
          Value codegenFunc(SrcOpT, ArrayRef<Type>, Value, OpBuilder *)>
class ElementWiseOpToLinalg : public OpConversionPattern<SrcOpT> {
public:
  using OpConversionPattern<SrcOpT>::OpConversionPattern;
  using OpAdaptor = typename SrcOpT::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ValueRange inputs = adaptor.getOperands();
    auto resultTy = cast<ShapedType>(op.getOutput().getType());
    Value output = getEmptyTensor(rewriter, loc, resultTy, {});

    int64_t maxRank = getMaxRank(adaptor.getOperands());

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(maxRank, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(maxRank);
    SmallVector<AffineMap> maps;
    for (Value v : inputs)
      maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);

    // Build `linalg.generic` op.
    bool failed = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? resultTy : TypeRange{}, inputs, output, maps,
        mlir::tosa::getNParallelLoopsAttrs(maxRank),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(output);
          Value innerResult =
              (*codegenFunc)(op, innerResultTy, args.front(), &rewriter);
          if (!innerResult) {
            failed = true;
          } else {
            nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
          }
        },
        linalg::getPrunedAttributeList(op));

    if (failed)
      return failure();

    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

using EluToLinalg = ElementWiseOpToLinalg<EluOp, mapEluOpToArithAndMathOps>;

struct ConvertXtenNNtoLinalg
    : public xilinx::xten::impl::ConvertXTenNNToLinalgBase<
          ConvertXtenNNtoLinalg> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, scf::SCFDialect, complex::ComplexDialect,
                math::MathDialect, shape::ShapeDialect, tensor::TensorDialect,
                arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    auto funcOp = getOperation();

    ConversionTarget target(*context);
    target.addIllegalOp<SignOp>();
    target.addLegalDialect<linalg::LinalgDialect, scf::SCFDialect,
                           complex::ComplexDialect, math::MathDialect,
                           shape::ShapeDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    RewritePatternSet patterns(context);
    patterns.add<EluToLinalg>(context);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace amd {
namespace xten_nn {

std::unique_ptr<mlir::Pass> createXTenNNToLinalgPass() {
  return std::make_unique<ConvertXtenNNtoLinalg>();
}

} // namespace xten_nn
} // namespace amd
