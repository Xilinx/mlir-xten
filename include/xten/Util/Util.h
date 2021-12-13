//===- Util.h ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace xilinx {
namespace xten {

Value MemRefTypeCast(OpBuilder &builder, Value val);

Value ToBuiltinTensorTypeCast(OpBuilder &builder, Value val);

Value FromTorchTensorTypeCast(OpBuilder &builder, Value val, Type resultTy);

Value TensorTypeCast(OpBuilder &builder, Value val, Type resultTy);

FuncOp getATenFn(ModuleOp module, std::string fnName, ArrayRef<Value> operands, ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);

uint64_t getTensorVolume(const Type ty);

}
}
