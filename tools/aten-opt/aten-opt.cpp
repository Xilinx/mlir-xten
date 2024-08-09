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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/InitAll.h"

#include "xten/Conversion/Passes.h"
#include "xten/Dialect/XTenNN/IR/XTenNN.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/Transforms/Passes.h"
#include "xten/Transform/Passes.h"

using namespace llvm;
using namespace mlir;

namespace mlir::test {
void registerTestConstantFold();
}

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::test::registerTestConstantFold();
  xilinx::xten::registerTransformPasses();
  xilinx::xten::registerConversionPasses();
  amd::xten_nn::registerXTenNNPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::registerAllDialects(registry);
  registry.insert<amd::xten_nn::XTenNNDialect,
                  torch::Torch::TorchDialect,
                  torch::TorchConversion::TorchConversionDialect>();

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry));
}
