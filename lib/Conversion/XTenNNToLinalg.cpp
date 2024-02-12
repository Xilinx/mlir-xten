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

inline Value getConstantOrSplat(OpBuilder *b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = t.dyn_cast<VectorType>()) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

// Adapted from the StableHLO to Linalg lowering in
// https://github.com/openxla/stablehlo/blob/main/stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h
Value mapSignOpToStdScalarOp(SignOp op, ArrayRef<Type> resultTypes,
                             Value operand, OpBuilder *b) {
  Location loc = op->getLoc();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(operand.getType()));
    Value ne0I1 = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, operand, zero);
    Value ne0Float =
        b->create<::mlir::arith::UIToFPOp>(loc, zero.getType(), ne0I1);
    Value copySign = b->create<::mlir::math::CopySignOp>(loc, resultTypes,
                                                         ne0Float, operand);
    auto isNan = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, operand, operand);
    return b->create<::mlir::arith::SelectOp>(loc, isNan, operand, copySign);
  }
  if (auto integerType = elementType.dyn_cast<IntegerType>()) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(operand.getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, operand.getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, operand.getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp = b->create<::mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 operand, zero);
    // This signed shift will either fill the integer with zeros or with ones,
    // depending on the sign. The or will then make sure that the positive case
    // returns a 1.
    Value ashr =
        b->create<::mlir::arith::ShRSIOp>(loc, operand, bitwidthMinusOne);
    Value orOp = b->create<::mlir::arith::OrIOp>(loc, ashr, one);
    // Check if the input was a 0
    return b->create<::mlir::arith::SelectOp>(loc, cmp, zero, orOp);
  }
  if (elementType.isa<ComplexType>()) {
    return b->create<::mlir::complex::SignOp>(loc, elementType, operand);
  }
  return nullptr;
}

// Mish(x) = x * tanh(Softplus(x, 1.0, 20.0))
//   where:
//     Softplus(x, b, th) = x * b > th ? x : log(1 + exp(x * b)) / b
//   therefore:
//     Mish(x) = x * tanh( (x > 20.0 ? x : log(1 + exp(x))) )
//
// Sources:
//  Mish decomposition:
//  https://github.com/llvm/torch-mlir/blob/main/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp#L4255
//  Softplus decomposition:
//  https://github.com/llvm/torch-mlir/blob/main/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp#L3255
Value mapMishOpToArithAndMathOps(MishOp op, ArrayRef<Type> /*resultTypes*/,
                                 Value operand, OpBuilder *b) {
  Location loc = op->getLoc();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (!isa<FloatType>(elementType)) {
    return nullptr;
  }

  // Build: log(1 + exp(x))
  Value exp = b->create<::mlir::math::ExpOp>(loc, operand);
  Value one =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 1));
  Value add = b->create<::mlir::arith::AddFOp>(loc, one, exp);
  Value log = b->create<::mlir::math::LogOp>(loc, add);

  // Build: x > 20.0 ? x : log(1 + exp(x))
  Value twenty =
      b->create<arith::ConstantOp>(loc, b->getFloatAttr(elementType, 20));
  Value cmpOp =
      b->create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, operand, twenty);
  Value softplus = b->create<arith::SelectOp>(loc, cmpOp, operand, log);

  // Finish: x * tanh(x > 20.0 ? x : log(1 + exp(x)))
  Value tanh = b->create<::mlir::math::TanhOp>(loc, softplus);
  return b->create<::mlir::arith::MulFOp>(loc, operand, tanh);
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

Value mapXTenNNRoundOpToRoundEvenOp(RoundOp op, ArrayRef<Type> /*resultTypes*/,
                                    Value operand, OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (!isa<FloatType>(elementType)) {
    return operand;
  }
  return b->create<::mlir::math::RoundEvenOp>(op->getLoc(), operand);
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
using MishToLinalg = ElementWiseOpToLinalg<MishOp, mapMishOpToArithAndMathOps>;
using RoundToLinalg =
    ElementWiseOpToLinalg<RoundOp, mapXTenNNRoundOpToRoundEvenOp>;
using SignToLinalg = ElementWiseOpToLinalg<SignOp, mapSignOpToStdScalarOp>;

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
    patterns.add<EluToLinalg, MishToLinalg, RoundToLinalg, SignToLinalg>(
        context);

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
