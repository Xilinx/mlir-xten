#include "ATenDialect.h"

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
  return Base::get(elemType.getContext(), ATenTypeKind::ATEN_LIST, elemType);
}

mlir::Type ATenListType::getElementType() {
  return getImpl()->getElementType();
}

mlir::Type ATenDialect::parseType(llvm::StringRef tyData,
                                  mlir::Location loc) const {

  llvm::StringRef tyDataOrig(tyData.str());

  if (tyData.startswith("list<")) {
    // extract the element type T from 'aten.list<T>'
    tyData = tyData.drop_front(StringRef("list<").size()).drop_back(1);

    // int
    if (tyData.startswith("i")) {
      unsigned int width = atoi(tyData.drop_front(1).str().c_str());
      return ATenListType::get(mlir::IntegerType::get(width, getContext()));
    }

    // float
    if (tyData.startswith("f")) {
      //int width = atoi(tyData.drop_front(1).str().c_str());
    }
  }

  emitError(loc, "Invalid ATen type '" + tyDataOrig + "'");
  return nullptr;
}

/// Print a ATenListType
void ATenDialect::printType(mlir::Type type, raw_ostream &os) const {
  auto ty = type.dyn_cast<ATenListType>();
  if (!ty) {
    os << "unknown aten type";
    return;
  }
  os << "list<";
  os << ty.getElementType();
  os << ">";
}

ATenDialect::ATenDialect(mlir::MLIRContext *ctx) : mlir::Dialect("aten", ctx) {
     addTypes<ATenListType>();
     addOperations<
#define GET_OP_LIST
#include "ATenOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "ATenOps.cpp.inc"

}
}