//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "xten/Dialect/XTenNN/IR/XTenNN.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterfaces.h"

using namespace mlir;
using namespace amd::xten_nn;

/// Parses a captured SSA operand.
///
/// Format:
///     ssa-id `=` ssa-id `:` type
static ParseResult parseCapture(OpAsmParser &p,
                                OpAsmParser::UnresolvedOperand &arg,
                                OpAsmParser::UnresolvedOperand &src,
                                Type &type) {
  // ssa-id `=` ssa-id `:` type
  if (p.parseOperand(arg))
    return failure();
  if (p.parseEqual())
    return failure();
  if (p.parseOperand(src))
    return failure();
  if (p.parseColon())
    return failure();
  if (p.parseType(type))
    return failure();

  return success();
}

/// Prints a captured SSA operand.
///
/// See parseCapture() for more details.
static void printCapture(OpAsmPrinter &p, Value arg, Value src) {
  p << arg << " = " << src << ": " << src.getType();
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
static void printCaptures(OpAsmPrinter &p, ValueRange args, ValueRange srcs) {
  auto srcIt = srcs.begin();
  p << '(';
  llvm::interleaveComma(args, p, [&](auto arg) {
    assert(srcIt != srcs.end());
    printCapture(p, arg, *srcIt++);
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
  if (p.parseRegion(*result.addRegion(), args, true))
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
  p << ' ';
  printCaptures(p, op.getEnclaveBody().getArguments(), op.getCaptures());
  p << ' ';

  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  if (!op->getAttrs().empty())
    p << ' ';

  p.printRegion(*op.getEnclaveBody().getParent(), false);

  if (op->getNumResults() > 0) {
    p << " -> ";
    interleaveComma(op->getResultTypes(), p);
  };
}

#define GET_OP_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

ParseResult SubgraphOp::parse(OpAsmParser &p, OperationState &result) {
  return parseEnclaveOp(p, result);
}

void SubgraphOp::print(OpAsmPrinter &p) {
  printEnclaveOp(p, *this);
}

LogicalResult SubgraphOp::verify() {
  // The number of captures must match the number of block arguments
  if (this->getCaptures().size() != this->getEnclaveBody().getNumArguments()) {
    return this->emitOpError()
           << "number of operands (" << this->getCaptures().size()
           << ") does not match number of arguments ("
           << this->getEnclaveBody().getNumArguments() << ")";
  }

  // The type of the arguments must match the types of the block arguments
  for (auto [idx, argType] :
       enumerate(this->getEnclaveBody().getArgumentTypes())) {
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

  if (dequantizeOp.getShift() != getShift())
    return {};

  auto dequantizeInput = dequantizeOp.getInput();
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

static std::string getResizeInvalidModeOption(ArrayRef<const char *> subOptions,
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

LogicalResult amd::xten_nn::ResizeOp::verify() {
  auto scales = getScales();
  if (scales.size() != 4) {
    return emitOpError("'" + getScalesAttrName().strref() +
                       "' must contain 4 values");
  }

  constexpr std::array coordinateTransformMode{"half_pixel", "asymmetric",
                                               "align_corners"};
  if (getCoordinateTransformationMode() > coordinateTransformMode.size() - 1) {
    return emitOpError(getResizeInvalidModeOption(
        coordinateTransformMode, getCoordinateTransformationModeAttrName()));
  }
  constexpr std::array mode{"Nearest", "Linear"};
  if (getMode() > mode.size() - 1) {
    return emitOpError(getResizeInvalidModeOption(mode, getModeAttrName()));
  }
  constexpr std::array nearestMode{"floor", "round_prefer_ceil",
                                   "round_prefer_floor"};
  if (getNearestMode() > nearestMode.size() - 1) {
    return emitOpError(
        getResizeInvalidModeOption(nearestMode, getNearestModeAttrName()));
  }

  return success();
}
