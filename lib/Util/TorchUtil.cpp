#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"


#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace xilinx {
namespace xten {

uint64_t getTensorVolume(const torch::Torch::BaseTensorType ty) {

  if (!ty.hasSizes())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getSizes())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(const Type ty) {
  if (auto t = ty.dyn_cast<torch::Torch::BaseTensorType>()) {
    return getTensorVolume(t);
  } else {
    return 1;
  }
}

/// Create a type cast to memref
Value MemRefTypeCast(OpBuilder &builder, Value val) {
  if (val.getType().isa<MemRefType>())
    return val;

  auto tensorTy = val.getType().dyn_cast<torch::Torch::BaseTensorType>();
  if (!tensorTy)
    return val; // error

  auto sizes = tensorTy.getSizes();
  auto dtype = tensorTy.getDtype();
  auto tensor = builder.create<torch::TorchConversion::ToBuiltinTensorOp>(
      val.getLoc(), RankedTensorType::get(sizes, dtype), val);
  auto memRefType = MemRefType::get(tensorTy.getSizes(), dtype, {}, 0);
  return builder.create<bufferization::ToMemrefOp>(val.getLoc(), memRefType, tensor)
      .getResult();
}

Value ToBuiltinTensorTypeCast(OpBuilder &builder, Value val) {
  if (val.getType().isa<MemRefType>())
    return val;

  auto tensorTy = val.getType().dyn_cast<torch::Torch::BaseTensorType>();
  if (!tensorTy)
    return val; // error

  auto sizes = tensorTy.getSizes();
  auto dtype = tensorTy.getDtype();
  return builder.create<torch::TorchConversion::ToBuiltinTensorOp>(
      val.getLoc(), RankedTensorType::get(sizes, dtype), val);
}

Value ToTorchTensorTypeCast(OpBuilder &builder, Value val, Type resultTy) {
  if (!val.getType().isa<TensorType>())
    return val;

  return builder.create<torch::TorchConversion::FromBuiltinTensorOp>(
      val.getLoc(), resultTy, val);
}

/// Create a type cast to tensor
Value TensorTypeCast(OpBuilder &builder, Value val, Type resultTy) {
  if (val.getType().isa<TensorType>())
    return val;
  auto refType = val.getType().dyn_cast<MemRefType>();
  if (!refType)
    return val;
  auto tensor =
      builder.create<bufferization::ToTensorOp>(val.getLoc(), val).getResult();
  return builder.create<torch::TorchConversion::FromBuiltinTensorOp>(
      val.getLoc(), resultTy, tensor);
}

}
}