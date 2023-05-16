//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_CONVERSION_PASSES
#define XTEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ATenToXTen : Pass<"aten-to-xten", "ModuleOp"> {
  let summary = "Convert ATen dialect to XTen dialect";
  let constructor = "xilinx::xten::createATenToXTenPass()";
}

def XTenToAffine : Pass<"xten-to-affine", "ModuleOp"> {
  let summary = "Convert XTen dialect to Affine dialect";
  let constructor = "xilinx::xten::createXTenToAffinePass()";
  let dependentDialects = ["mlir::AffineDialect"];
}

def XTenToLinalg : Pass<"xten-to-linalg", "ModuleOp"> {
  let summary = "Convert XTen dialect to Linalg dialect";
  let constructor = "xilinx::xten::createXTenToLinalgPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect",
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
  ];
}

def TOSAToXTenNN : Pass<"tosa-to-xten-nn", "ModuleOp"> {
  let summary = "Partial Conversion of TOSA to XTenNN.";
  let description = [{Converts all TOSA operators that match to the quantization
  operators in XTenNN.
  }];
  let constructor = "amd::xten_nn::createTOSAToXTenNNPass()";  
  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "amd::xten_nn::XTenNNDialect"
  ];
}

def XTenNNToTosa : Pass<"xten-nn-to-tosa", "ModuleOp"> {
  let summary = "Convert XTenNN quantization operations to TOSA.";
  let description = [{
  Partial conversion of XTenNN operations to TOSA.

  This pass only converts XTenNN quantize and dequantize operations 
  who operate on signed integer tensors to TOSA.

  Currently, unsigned integer tensor support in TOSA does not allow
  lowerings of operations like:
  ```
  xten_nn.quantize(%arg0 : tensor<1x2xf32>){shift = -3 : si32} -> tensor<1x2xui8>
  ```
  to TOSA. This pass will ignore any quantize or dequantize operations
  working on unsigned tensors for the moment.
  }];
  let constructor = "amd::xten_nn::createXTenNNToTOSAPass()";
  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "amd::xten_nn::XTenNNDialect"
  ];
}

#endif
