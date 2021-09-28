
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#define DEBUG_TYPE "xten-util"

using namespace mlir;

namespace xilinx {
namespace xten {

namespace {
  
std::string getMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
    ret << "M";
    auto shape = mrt.getShape();
    const Type elem = mrt.getElementType();
    for (auto s : shape)
      ret << s << "x";
    ret << getMangledType(elem);
  }
  else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  }
  else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  }
  else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
    ret << "I64";
  }
  else if (const NPCOMP::aten::ATenListType alt = ty.dyn_cast<const NPCOMP::aten::ATenListType>()) {

  }
  else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

std::string getMangledFuncName(ModuleOp module, std::string prefix, FunctionType fnTy) {
  std::string sep = "_";

  auto resultTy = fnTy.getResults();
  auto operTy = fnTy.getInputs();

  std::string ret = prefix;
  for (const Type t : resultTy)
    ret = ret + sep + getMangledType(t);
  for (const Type t : operTy)
    ret = ret + sep + getMangledType(t);

  return ret;
}
}

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value> operands, ArrayRef<Type> retTys)
{
  Builder builder(module);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix+"_AtenAcapOp", fnTy);
  auto fn = module.lookupSymbol<FuncOp>(fnName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  return fn;
}

uint64_t getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(const Type ty) {
  if (auto t = ty.dyn_cast<ShapedType>()) {
    return getTensorVolume(t);
  }
  else {
    return 1;
  }
}

}
}