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
