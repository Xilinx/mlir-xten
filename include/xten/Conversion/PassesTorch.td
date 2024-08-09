//===- PassesTorch.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTEN_CONVERSION_PASSES_TORCH
#define XTEN_CONVERSION_PASSES_TORCH

include "mlir/Pass/PassBase.td"

def ConvertXTenNNToTorch : Pass<"convert-xtennn-to-torch", "mlir::func::FuncOp"> {
  let summary = "Convert XTenNN operations to Torch.";
  let description = [{
    Partial conversion of XTenNN operations to Torch.
  }];
  let constructor = "amd::xten_nn::createXTenNNToTorchPass()";
  let dependentDialects = [
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
  ];
}

#endif // XTEN_CONVERSION_PASSES_TORCH
