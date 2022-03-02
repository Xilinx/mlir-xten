//===- Util.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
  } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  } else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
    ret << "I64";
  } else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

std::string getMangledFuncName(ModuleOp module, std::string prefix,
                               FunctionType fnTy) {
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
} // namespace

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

FuncOp getATenFn(ModuleOp module, std::string prefix, ArrayRef<Value> operands,
                 ArrayRef<Type> retTys) {
  Builder builder(module);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix + "_AtenAcapOp", fnTy);
  auto fn = module.lookupSymbol<FuncOp>(fnName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  return fn;
}

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

} // namespace xten
} // namespace xilinx
