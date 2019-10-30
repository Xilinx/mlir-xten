//===- ATenDialect.h - Dialect definition for the ATen IR ----------------===//
//
// Copyright 2019 Xilinx
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_ATEN_DIALECT_H
#define MLIR_ATEN_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

using namespace mlir;

namespace xilinx {
namespace aten {

// The Dialect
class ATenDialect : public mlir::Dialect {
public:
  explicit ATenDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "aten"; }


  /// Parse a type registered to this dialect. Overridding this method is
  /// required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  mlir::Type parseType(llvm::StringRef tyData,
                       mlir::Location loc) const override;

  /// Print a type registered to this dialect. Overridding this method is
  /// only required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  void printType(mlir::Type type, llvm::raw_ostream &os) const override;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {
struct ATenListTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum ATenTypeKind {
  // The enum starts at the range reserved for this dialect.
  ATEN_TYPE = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  ATEN_LIST,
};

/// Type for Toy arrays.
/// In MLIR Types are reference to immutable and uniqued objects owned by the
/// MLIRContext. As such `ATenListType` only wraps a pointer to an uniqued
/// instance of `ATenListTypeStorage` (defined in our implementation file) and
/// provides the public facade API to interact with the type.
class ATenListType : public mlir::Type::TypeBase<ATenListType, mlir::Type,
                                                 detail::ATenListTypeStorage> {
public:
  using Base::Base;

  /// Return the type of individual elements in the array.
  mlir::Type getElementType();

  /// Get the unique instance of this Type from the context.
  static ATenListType get(mlir::Type elementType);

  /// Support method to enable LLVM-style RTTI type casting.
  static bool kindof(unsigned kind) { return kind == ATenTypeKind::ATEN_LIST; }
};


////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "ATenOps.h.inc"

}
}

#endif