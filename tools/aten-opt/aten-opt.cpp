//===- aten-opt.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2019 - 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "xten/Conversion/Passes.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  xilinx::xten::registerConversionPasses();
  amd::xten_nn::registerXTenNNPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::registerAllDialects(registry);
  registry.insert<amd::xten_nn::XTenNNDialect>();

  return failed(
      MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}
