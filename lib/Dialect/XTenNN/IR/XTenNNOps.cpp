//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "xten/Dialect/XTenNN/IR/XTenNN.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterfaces.h"

#include <cstdint>

using namespace mlir;
using namespace amd::xten_nn;

/// Parses a captured SSA operand.
///
/// Format:
///     One of:
///       ssa-id `=` ssa-id `:` type
///       ssa-id `:` type
static ParseResult parseCapture(OpAsmParser &p,
                                OpAsmParser::UnresolvedOperand &arg,
                                OpAsmParser::UnresolvedOperand &src,
                                Type &type) {
  if (p.parseOperand(arg))
    return failure();
  if (failed(p.parseOptionalEqual())) {
    src = arg;
    arg = {};
  } else {
    if (p.parseOperand(src))
      return failure();
  }
  if (p.parseColon())
    return failure();
  if (p.parseType(type))
    return failure();

  return success();
}

/// Prints a captured SSA operand.
///
/// See parseCapture() for more details.
static void printCapture(OpAsmPrinter &p, Value src) {
  p << src << ": " << src.getType();
}

/// Prints a captured SSA operand.
///
/// See parseCapture() for more details.
static void printCapture(OpAsmPrinter &p, Value arg, Value src) {
  p << arg << " = ";
  printCapture(p, src);
}

/// Parses a comma-separated list of zero or more captured SSA operands.
///
/// Format:
///     `(` [ capture { `,` capture } ] `)`
static ParseResult parseCaptures(OpAsmParser &p,
                                 SmallVectorImpl<OpAsmParser::Argument> &args,
                                 SmallVectorImpl<Value> &srcs) {
  // `(` [ capture { `,` capture } ] `)`
  return p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        auto &arg = args.emplace_back();
        OpAsmParser::UnresolvedOperand src;
        if (parseCapture(p, arg.ssaName, src, arg.type))
          return failure();
        if (p.resolveOperand(src, arg.type, srcs))
          return failure();
        return success();
      });
}

/// Prints a comma-separated list of zero or more captured SSA operands.
///
/// See parseCaptures() for more details.
static void printCaptures(OpAsmPrinter &p, ValueRange srcs) {
  p << '(';
  llvm::interleaveComma(srcs, p, [&](auto src) { printCapture(p, src); });
  p << ')';
}

/// Prints a comma-separated list of zero or more captured SSA operands.
///
/// See parseCaptures() for more details.
static void printCaptures(OpAsmPrinter &p, ValueRange args, ValueRange srcs) {
  auto argIt = args.begin();
  p << '(';
  llvm::interleaveComma(srcs, p, [&](auto src) {
    assert(argIt != args.end());
    printCapture(p, *argIt++, src);
  });
  p << ')';
}

/// Parses a trivial EnclaveOp.
///
/// Format:
///     capture-list [ attr-dict-with-keyword ] region [ `->` type-list ]
static ParseResult parseEnclaveOp(OpAsmParser &p, OperationState &result) {
  // `(` captures `)`
  SmallVector<OpAsmParser::Argument> args;
  if (parseCaptures(p, args, result.operands))
    return failure();

  // [ attr-dict-with-keyword ]
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // `{` ... `}`
  auto &region = *result.addRegion();
  auto parseResult = p.parseOptionalRegion(region, args, true);
  if (parseResult.has_value() && failed(*parseResult))
    return failure();

  // [ `->` type-list ]
  if (succeeded(p.parseOptionalArrow())) {
    if (p.parseTypeList(result.types))
      return failure();
  }

  return success();
}

/// Prints a trivial EnclaveOp.
///
/// See parseEnclaveOp() for more details.
static void printEnclaveOp(OpAsmPrinter &p, EnclaveOp op) {
  Block *optBody = op.getOptionalEnclaveBody();
  p << ' ';
  if (optBody) {
    printCaptures(p, optBody->getArguments(), op.getCaptures());
  } else {
    printCaptures(p, op.getCaptures());
  }
  p << ' ';

  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  if (!op->getAttrs().empty())
    p << ' ';

  if (optBody) {
    p.printRegion(*optBody->getParent(), false);
    p << ' ';
  }

  if (op->getNumResults() > 0) {
    p << "-> ";
    interleaveComma(op->getResultTypes(), p);
  };
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

/// Parses a list of ssa values with their types.
/// `(` (ssa-id `:` type (`,` ssa-id `:` type)*)? `)`
///
/// This method is used by the tablegen assembly format for the kernel op.
static ParseResult parseKernelArgumentList(OpAsmParser &p,
                                           SmallVectorImpl<Value> &operands) {
  return p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren,
      [&]() -> ParseResult {
        OpAsmParser::UnresolvedOperand operand;
        Type type;
        if (p.parseOperand(operand))
          return failure();
        if (p.parseOptionalColon())
          return p.emitError(p.getCurrentLocation(),
                             "expected ':`, (argument format is val : type)");

        if (p.parseType(type) || p.resolveOperand(operand, type, operands))
          return failure();
        return success();
      },
      " in argument list");
}

/// Prints a list of ssa values with their types.
/// `(` (ssa-id `:` type (`,` ssa-id `:` type)*)? `)`
///
/// This method is used by the tablegen assembly format for the kernel op.
static void printKernelArgumentList(OpAsmPrinter &p, TypeRange types,
                                    OperandRange arguments) {
  p << "(";
  llvm::interleaveComma(llvm::zip(arguments, types), p, [&](const auto &a) {
    p << get<0>(a) << " : " << get<1>(a);
  });
  p << ")";
}

// Parse
// {((name = )?value, )*((name = )?value)}
static ParseResult parseKernelInstantiationArgs(OpAsmParser &p,
                                                SmallVector<Attribute> &values,
                                                SmallVector<Attribute> &names) {
  if (failed(p.parseLBrace()))
    return failure();

  if (failed(p.parseCommaSeparatedList([&p, &names, &values]() {
        std::string name;
        bool hasName = false;
        if (succeeded(p.parseOptionalKeywordOrString(&name))) {
          hasName = true;
          if (failed(p.parseEqual()))
            return failure();
        }
        Attribute attr;
        auto res = p.parseOptionalAttribute(attr);
        if (res.has_value() && succeeded(*res)) {
          if (hasName)
            names.push_back(StringAttr::get(p.getContext(), name));
          values.push_back(attr);
        }
        if (res.has_value() && failed(*res))
          return failure();

        return success();
      }))) {
    return failure();
  }

  if (failed(p.parseRBrace()))
    return failure();

  return success();
}

// Print
// instantiation_args {((name = )?value, )*((name = )?value)}
static void
printKernelInstantiationArgs(OpAsmPrinter &p,
                             ArrayRef<Attribute> instantiationArgs,
                             ArrayRef<Attribute> instantiationArgNames) {
  if (!instantiationArgs.empty()) {
    p << "instantiation_args {";
    auto zipped = llvm::zip_longest(instantiationArgNames, instantiationArgs);
    for (auto iter = zipped.begin(); iter != zipped.end(); ++iter) {
      if (iter != zipped.begin())
        p << ", ";
      auto [name, value] = *iter;
      if (name)
        p << *name << " = ";
      if (value)
        p.printAttribute(*value);
    }
    p << '}';
  }
}

// Parse
//  $name custom<KernelArgumentList>(type($arguments), $arguments)
//  (instantiation_args custom<InstantiationArgs>)? attr-dict
//  `->` type($results)
ParseResult KernelOp::parse(OpAsmParser &p, OperationState &result) {
  StringAttr name;
  if (p.parseAttribute(name, "name", result.attributes))
    return failure();

  if (parseKernelArgumentList(p, result.operands))
    return failure();

  if (succeeded(p.parseOptionalKeyword("instantiation_args"))) {
    SmallVector<Attribute> values;
    SmallVector<Attribute> names;
    if (failed(parseKernelInstantiationArgs(p, values, names)))
      return failure();
    result.addAttribute("instantiation_args",
                        ArrayAttr::get(p.getContext(), values));
    if (!names.empty()) {
      result.addAttribute("instantiation_arg_names",
                          ArrayAttr::get(p.getContext(), names));
    }
  }

  if (p.parseOptionalAttrDict(result.attributes))
    return failure();

  // If the op has no results, the `-> type($results)` is absent.
  if (p.parseOptionalArrow())
    return success();

  if (p.parseTypeList(result.types))
    return failure();

  return success();
}

// Print
//  $name custom<KernelArgumentList>(type($arguments), $arguments)
//  (instantiation_args custom<InstantiationArgs>)? attr-dict
//  `->` type($results)
void KernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getNameAttr();
  p << ' ';
  printKernelArgumentList(p, getOperandTypes(), getOperands());
  p << ' ';
  auto instantiationArgs = getInstantiationArgs();
  auto instantiationArgNames = getInstantiationArgNames();
  if (instantiationArgs != std::nullopt) {
    printKernelInstantiationArgs(p, instantiationArgs->getValue(),
                                 (instantiationArgNames == std::nullopt)
                                     ? ArrayRef<Attribute>()
                                     : instantiationArgNames->getValue());
    p << ' ';
  }

  SmallVector<StringRef> elidedAttrs = {"name", "instantiation_args",
                                        "instantiation_arg_names"};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
  if (llvm::any_of(
          getOperation()->getAttrs(), [&elidedAttrs](NamedAttribute a) {
            auto name = a.getName();
            return llvm::any_of(elidedAttrs, [&name](StringRef elidedName) {
              return name == elidedName;
            });
          }))
    p << ' ';
  if (getNumResults()) {
    p << "-> ";
    p << getResultTypes();
  }
}

LogicalResult KernelOp::verify() {
  if (getInstantiationArgNames().has_value()) {
    if (!getInstantiationArgs().has_value())
      return emitOpError(
          "cannot have instantiation arg names without instantiation args");
    if (!(getInstantiationArgNamesAttr().empty() ||
          getInstantiationArgNamesAttr().size() ==
              getInstantiationArgsAttr().size()))
      return emitOpError("instantiation arg names must be either empty or as "
                         "long as instantiation args");
  }

  return success();
}

#define GET_OP_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

ParseResult SubgraphOp::parse(OpAsmParser &p, OperationState &result) {
  return parseEnclaveOp(p, result);
}

void SubgraphOp::print(OpAsmPrinter &p) { printEnclaveOp(p, *this); }

LogicalResult SubgraphOp::verify() {
  Block *optBody = this->getOptionalEnclaveBody();
  if (!optBody) {
    // Nothing to verify
    return success();
  }

  // The number of captures must match the number of block arguments
  if (this->getCaptures().size() != optBody->getNumArguments()) {
    return this->emitOpError()
           << "number of operands (" << this->getCaptures().size()
           << ") does not match number of arguments ("
           << optBody->getNumArguments() << ")";
  }

  // The type of the arguments must match the types of the block arguments
  for (auto [idx, argType] : enumerate(optBody->getArgumentTypes())) {
    if (this->getCapture(idx).getType() != argType) {
      return this->emitOpError()
             << "type of operand #" << idx << " ("
             << this->getCapture(idx).getType()
             << ") does not match argument type (" << argType << ")";
    }
  }
  return success();
}

LogicalResult SubgraphOp::inferReturnTypeComponents(
    MLIRContext * /*context*/, ::std::optional<Location> /*location*/,
    ValueShapeRange /*operands*/, DictionaryAttr /*attributes*/,
    OpaqueProperties /*properties*/, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  llvm::SmallVector<ShapedTypeComponents, 2> returnShapes;

  // TODO: Ideally, we would walk over the operations in the
  // subgraph region and have their shapes also inferred by
  // the InferShapedTypeOpInterface. However, this is enough
  // for our shape inference because our walk is performed by
  // another pass. We may need to extend this later if it needs
  // generalization.
  Operation *terminator = regions.front()->front().getTerminator();
  for (Type type : terminator->getOperandTypes()) {
    auto shapedType = llvm::dyn_cast<ShapedType>(type);

    if (!shapedType)
      return failure();

    returnShapes.push_back(shapedType);
  }
  inferredReturnShapes.append(returnShapes);
  return success();
}

//===----------------------------------------------------------------------===//
// XTenNNDialect
//===----------------------------------------------------------------------===//

OpFoldResult amd::xten_nn::QuantizeOp::fold(FoldAdaptor adaptor) {
  // Fold away cases where a xten_nn.quantize is preceeded by xten_nn.dequantize
  // that uses the same shift factor and has same types.

  auto dequantizeOp =
      dyn_cast_or_null<amd::xten_nn::DequantizeOp>(getInput().getDefiningOp());
  if (!dequantizeOp)
    return {};

  if (!dequantizeOp->hasOneUse() || dequantizeOp.getShift() != getShift())
    return {};

  auto dequantizeInput = dequantizeOp.getInput();
  if (dequantizeInput.getType() != getType())
    return {};

  return dequantizeInput;
}

OpFoldResult amd::xten_nn::GroupQuantizeOp::fold(FoldAdaptor adaptor) {
  // Fold away cases where a xten_nn.group_quantize is preceeded by
  // xten_nn.group_dequantize that uses the same shift factor and has same
  // types.

  auto dequantizeOp = dyn_cast_or_null<amd::xten_nn::GroupDequantizeOp>(
      getInput().getDefiningOp());
  if (!dequantizeOp)
    return {};

  if (!dequantizeOp->hasOneUse())
    return {};
  if (dequantizeOp.getScales() != getScales())
    return {};
  if (dequantizeOp.getZeros() != getZeros())
    return {};

  auto dequantizeInput = dequantizeOp.getQuants();
  if (dequantizeInput.getType() != getType())
    return {};

  return dequantizeInput;
}

void amd::xten_nn::XTenNNDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"
      >();
}

LogicalResult amd::xten_nn::GroupConv2dOp::verify() {
  auto inputShape = cast<ShapedType>(getInput().getType()).getShape();
  auto weightShape = cast<ShapedType>(getWeights().getType()).getShape();
  const auto group = getGroup();

  if (inputShape[3] == static_cast<int64_t>(group) &&
      weightShape[1] == static_cast<int64_t>(group)) {
    return emitOpError(
        "groups needs to be different than input and output channel");
  }

  if (group < 1) {
    return emitOpError("groups expected to be at least one");
  }

  auto pads = getPad().getValue();
  auto firstDenseI64Array = dyn_cast<DenseI64ArrayAttr>(pads[0]);
  auto secondDenseI64Array = dyn_cast<DenseI64ArrayAttr>(pads[1]);
  if (!firstDenseI64Array || !secondDenseI64Array ||
      firstDenseI64Array.size() != 2 || secondDenseI64Array.size() != 2) {
    return emitOpError(
        "pad attribute expected to be a 2x2 i64 array. Eg: [[0, 1], [1, 0]]");
  }

  return success();
}

LogicalResult amd::xten_nn::GroupQuantizeOp::verify() {
  auto inputShape = cast<ShapedType>(getInput().getType()).getShape();
  auto scalesShape = cast<ShapedType>(getScales().getType()).getShape();
  auto zerosShape = cast<ShapedType>(getZeros().getType()).getShape();
  auto quantsShape = cast<ShapedType>(getQuants().getType()).getShape();

  if (inputShape != quantsShape) {
    return emitOpError() << "input and quants must have the same shape ("
                         << inputShape << " v " << quantsShape << ")";
  }

  if (scalesShape != zerosShape) {
    return emitOpError() << "scales and zeros must have the same shape ("
                         << scalesShape << " v " << zerosShape << ")";
  }

  if (scalesShape.back() != 1) {
    return emitOpError() << "groups needs to be expressed in the innermost "
                            "dimension of scales vs quants ("
                         << scalesShape.back() << ")";
  }

  if (scalesShape.drop_back() != quantsShape.drop_back()) {
    return emitOpError() << "scales and quants must have the same shape except "
                            "for the innermost dimension ("
                         << scalesShape << " v " << quantsShape << ")";
  }

  // TODO validate:
  // - bits can contain range [min, max].
  // - quant dtype is at least bits wide.

  return success();
}

LogicalResult amd::xten_nn::GroupDequantizeOp::verify() {
  auto outputShape = cast<ShapedType>(getOutput().getType()).getShape();
  auto scalesShape = cast<ShapedType>(getScales().getType()).getShape();
  auto zerosShape = cast<ShapedType>(getZeros().getType()).getShape();
  auto quantsShape = cast<ShapedType>(getQuants().getType()).getShape();

  if (outputShape != quantsShape) {
    return emitOpError() << "output and quants must have the same shape ("
                         << outputShape << " v " << quantsShape << ")";
  }

  if (scalesShape != zerosShape) {
    return emitOpError() << "scales and zeros must have the same shape ("
                         << scalesShape << " v " << zerosShape << ")";
  }

  if (scalesShape.back() != 1) {
    return emitOpError() << "groups needs to be expressed in the innermost "
                            "dimension of scales vs quants ("
                         << scalesShape.back() << ")";
  }

  if (scalesShape.drop_back() != quantsShape.drop_back()) {
    return emitOpError() << "scales and quants must have the same shape except "
                            "for the innermost dimension ("
                         << scalesShape << " v " << quantsShape << ")";
  }

  // TODO validate:
  // - bits can contain range [min, max].
  // - quant dtype is at least bits wide.

  return success();
}

static std::string getOpInvalidModeOption(ArrayRef<const char *> subOptions,
                                          StringRef option) {
  std::string result;
  llvm::raw_string_ostream rso(result);

  unsigned idx = 0;
  llvm::interleaveComma(subOptions, rso, [&](StringRef option) {
    rso << llvm::formatv("'{0}'({1})", option, idx++).str();
  });

  return llvm::formatv("Valid values for '{0}' option are: {1}", option,
                       rso.str())
      .str();
}

LogicalResult amd::xten_nn::GridSampleOp::verify() {

  constexpr std::array mode{"bilinear"};
  if (getMode() > mode.size() - 1) {
    return emitOpError(getOpInvalidModeOption(mode, getModeAttrName()));
  }
  constexpr std::array paddingMode{"zeros", "border"};
  if (getPaddingMode() > paddingMode.size() - 1) {
    return emitOpError(
        getOpInvalidModeOption(paddingMode, getPaddingModeAttrName()));
  }

  return success();
}

LogicalResult amd::xten_nn::ResizeOp::verify() {
  auto scales = getScales();
  if (scales.size() != 4) {
    return emitOpError("'" + getScalesAttrName().strref() +
                       "' must contain 4 values");
  }

  constexpr std::array coordinateTransformMode{
      "half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"};
  if (getCoordinateTransformationMode() > coordinateTransformMode.size() - 1) {
    return emitOpError(getOpInvalidModeOption(
        coordinateTransformMode, getCoordinateTransformationModeAttrName()));
  }
  constexpr std::array mode{"Nearest", "Linear"};
  if (getMode() > mode.size() - 1) {
    return emitOpError(getOpInvalidModeOption(mode, getModeAttrName()));
  }
  constexpr std::array nearestMode{"floor", "round_prefer_ceil",
                                   "round_prefer_floor"};
  if (getNearestMode() > nearestMode.size() - 1) {
    return emitOpError(
        getOpInvalidModeOption(nearestMode, getNearestModeAttrName()));
  }

  return success();
}

std::optional<uint64_t> getConstantK(Value k) {
  auto *op = k.getDefiningOp();
  if (!op) {
    return {};
  }
  auto constantOp = dyn_cast<arith::ConstantOp>(op);
  if (!constantOp)
    return {};
  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
  if (!intAttr)
    return {};
  return (uint64_t)
      intAttr.getInt(); // Always positive by definition of onnx.topk
}

LogicalResult TopK::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    TopK::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());

  auto axis = (int64_t)adaptor.getAxis();
  // onnx spec: axis: [-r, r-1]
  if (axis < -inTy.getRank() || axis >= inTy.getRank()) {
    return emitOptionalError(location,
                             "expected axis to be within [-rank,rank) (where "
                             "rank is the rank of the input)");
  }

  // normalize axis: [0, r)
  if (axis < 0) {
    axis += inTy.getRank();
  }

  assert((axis >= 0 && axis < inTy.getRank()) && "axis has invalid value");

  auto dimSize = inTy.getDimSize(axis);
  auto k = getConstantK(adaptor.getK());
  // If both k and dim are known statically, we can check that k <= dim
  if (k && dimSize != ShapedType::kDynamic) {
    if ((uint64_t)dimSize <= *k) {
      return emitOptionalError(location, "expected k <= dimension size");
    }
  }

  SmallVector<int64_t> resultShape{inTy.getShape()};
  resultShape[axis] = k ? *k : ShapedType::kDynamic;

  inferredReturnShapes.push_back(
      ShapedTypeComponents(resultShape, inTy.getElementType()));
  inferredReturnShapes.push_back(
      ShapedTypeComponents(resultShape, IntegerType::get(context, 64)));
  return success();
}

bool TopK::isCompatibleReturnTypes(mlir::TypeRange l, mlir::TypeRange r) {
  if (l.size() != r.size() || l.size() != 2)
    return false;

  auto sameElementType =
      getElementTypeOrSelf(l[0]) == getElementTypeOrSelf(r[0]) &&
      getElementTypeOrSelf(l[1]) == getElementTypeOrSelf(r[1]);
  return sameElementType && succeeded(verifyCompatibleShapes(l, r));
}

LogicalResult ReduceMeanOp::inferReturnTypeComponents(
    MLIRContext * /*context*/, std::optional<Location> location,
    ReduceMeanOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
  auto keepDims = adaptor.getKeepdims();
  auto axes = adaptor.getAxes();

  // Sanitize axes
  llvm::SmallVector<int64_t> newAxes;
  for (auto axis : axes) {
    // onnx spec: axis: [-r, r-1]
    if (axis < -inTy.getRank() || axis >= inTy.getRank()) {
      return emitOptionalError(location,
                               "expected axis to be within [-rank,rank) (where "
                               "rank is the rank of the input)");
    }

    // normalize axis: [0, r)
    if (axis < 0) {
      axis += inTy.getRank();
    }

    assert((axis >= 0 && axis < inTy.getRank()) && "axis has invalid value");
    newAxes.push_back(axis);
  }

  SmallVector<int64_t, 4> outputShape;
  auto inputShape = inTy.getShape();
  for (auto [idx, dim] : llvm::enumerate(inputShape)) {
    if (llvm::is_contained(axes, idx)) {
      if (keepDims) {
        outputShape.push_back(1);
      }
    } else {
      outputShape.push_back(dim);
    }
  }

  inferredReturnShapes.push_back(
      ShapedTypeComponents(outputShape, inTy.getElementType()));
  return success();
}