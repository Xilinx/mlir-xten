// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "ATenDialect.h"
#include "ATenPasses.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace xilinx {
namespace aten {

namespace detail {

/// This class holds the implementation of the ATenListType.
/// It is intended to be uniqued based on its content and owned by the context.
struct ATenListTypeStorage : public mlir::TypeStorage {
  ATenListTypeStorage(Type elementType) : elementType(elementType) {}

  /// The hash key used for uniquing.
  using KeyTy = mlir::Type;
  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  /// This is a factory method to create our type storage. It is only
  /// invoked after looking up the type in the context using the key and not
  /// finding it.
  static ATenListTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {

    // Allocate the instance for the ATenListTypeStorage itself
    auto *storage = allocator.allocate<ATenListTypeStorage>();
    // Initialize the instance using placement new.
    return new (storage) ATenListTypeStorage(key);
  }

  Type getElementType() const { return elementType; }

private:
  Type elementType;

};
} // namespace detail

ATenListType ATenListType::get(mlir::Type elemType) {
  return Base::get(elemType.getContext(), elemType);
}

mlir::Type ATenListType::getElementType() {
  return getImpl()->getElementType();
}

mlir::Type ATenDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  // All types start with an identifier that we switch on.
  StringRef typeNameSpelling;
  if (failed(parser.parseKeyword(&typeNameSpelling)))
    return nullptr;

  if (typeNameSpelling == "list") {
    if(failed(parser.parseLess()))
      return nullptr;
    Type t;
    if(failed(parser.parseType(t)))
      return nullptr;
    if(failed(parser.parseGreater()))
      return nullptr;
    return ATenListType::get(t);
  }

  parser.emitError(parser.getCurrentLocation(), "Invalid ATen type '" + typeNameSpelling + "'");
  return nullptr;
}

/// Print a ATenListType
void ATenDialect::printType(mlir::Type type, DialectAsmPrinter &os) const {
  auto ty = type.dyn_cast<ATenListType>();
  if (!ty) {
    os << "unknown aten type";
    return;
  }
  os << "list<";
  os.getStream() << ty.getElementType();
  os << ">";
}

ATenDialect::ATenDialect(mlir::MLIRContext *ctx) :
  mlir::Dialect("aten", ctx, ::mlir::TypeID::get<ATenDialect>()) {
     addTypes<ATenListType>();
     addOperations<
#define GET_OP_LIST
#include "ATen.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "ATen.cpp.inc"

#include "ATenOpInterfaces.cpp.inc"

void registerATenPasses() {
// #define GEN_PASS_REGISTRATION
//   #include "ATenPasses.h.inc"
  xilinx::aten::registerAcapHerdAssignPass();
  xilinx::aten::registerAcapHerdLoweringPass();
  xilinx::aten::registerAcapLoopLoweringPass();
  xilinx::aten::registerAffineLoopOptPass();
  xilinx::aten::registerATenAcapFusionPass();
  xilinx::aten::registerATenLayerNamePass();
  xilinx::aten::registerATenLoweringPass();
  xilinx::aten::registerATenSimpleAllocatePass();
  xilinx::aten::registerReturnEliminationPass();

}


} // namespace aten
} // namespace xilinx
