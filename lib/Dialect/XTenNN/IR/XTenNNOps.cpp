//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#include "mlir/Support/LogicalResult.h"
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

//===----------------------------------------------------------------------===//
// XTenNNDialect
//===----------------------------------------------------------------------===//

void amd::xten_nn::XTenNNDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "xten/Dialect/XTenNN/IR/XTenNNOps.cpp.inc"
      >();
}