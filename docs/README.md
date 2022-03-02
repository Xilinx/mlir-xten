# Extensions to Torch-MLIR

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains Xilinx extensions to the torch-mlir ATen dialect to enable expressing the input and output of partial results. Using these extensions, a prototype streaming dataflow exploration tool scans the design space for compute/memory/computation balanced partitioning of CNNs on the AIE array of Xilinx' Versal devices.

[Getting Started](Building.md)

[Github sources](https://github.com/Xilinx/mlir-xten)

Generated code documentation
- [XTen Dialect](XTenDialect.md)
- XTen Passes
    - [Dialect Passes](XTenDialectPasses.md)
    - [Transform Passes](XTenTransformPasses.md)
    - [Conversion Passes](XTenConversionPasses.md)

-----

<p align="center">Copyright&copy; 2019-2022 Xilinx</p>
