# mlir-xten

This repository contains an extension of the npcomp ATEN dialect to enable expressing the input and output of partial results. 
Using these extensions, a prototype streaming dataflow exploration tool scans the desing space for compute/memory/computation balanced partitionings of CNNs on the AIE array of Xilinx' Versal devices.

## Building the MLIR XTEN toolchain

### Prerequisites

### Building on X86

```sh
git clone https://github.com/Xilinx/cmakeModules
git clone https://gitenterprise.xilinx.com/XRLabs/mlir-xten.git
mkdir build; cd build
cmake ..\
    -DLLVM_DIR=${absolute path to LLVMBUILD}/lib/cmake/llvm \
    -DMLIR_DIR=${absolute path to LLVMBUILD}/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=/absolute/path/to/cmakeModules/ \
    -DNPCOMP_SOURCE_DIR={/absolute/path/to/npcomp}/mlir-npcomp
    -DNPCOMP_BINARY_DIR={absolute path to npcomp BUILD}
```

## Environment setup

-----
<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
