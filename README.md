# mlir-xten

This repository contains an extension of the npcomp ATEN dialect to enable expressing the input and output of partial results. 
Using these extensions, a prototype streaming dataflow exploration tool scans the design space for compute/memory/computation balanced partitionings of CNNs on the AIE array of Xilinx' Versal devices.

## Building the MLIR XTEN toolchain

### Prerequisites

```
cmake 3.17.5 or higher
clang/llvm 13+ from source https://github.com/llvm/llvm-project
llvm/torch-mlir from https://github.com/llvm/torch-mlir
```

### Building on X86

NOTE: this initial version of mlir-xten currently fails to build against head torch-mlir. Fixes in progress.

#### First Compile torch-mlir

Clone torch-mlir from https://github.com/llvm/torch-mlir and follow the build instructions.
Make sure to add following extra CMake flags: -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON
and build everything

```sh
cmake --build build
```

#### Finally compile mlir-xten

```sh
git clone https://github.com/Xilinx/mlir-xten.git
mkdir build; cd build
cmake ..\
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DLLVM_DIR=${absolute path to torchMlirBuild}/cmake/llvm \
    -DMLIR_DIR=${absolute path to torchMlirBuild}/cmake/mlir \
    -DTORCH_MLIR_SOURCE_DIR=${absolute path to torchMlirSource} \
    -DTORCH_MLIR_BINARY_DIR=${absolute path to torchMlirBuild}/tools/torch-mlir
```

## Environment setup

TBD

-----
<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
