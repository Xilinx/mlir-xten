# Building the code

## Prerequisites

```
cmake 3.20.6 or higher
ninja 1.8.2
clang/llvm 14+ from source https://github.com/llvm/llvm-project
llvm/torch-mlir from https://github.com/llvm/torch-mlir
```

## Building on X86

NOTE: this initial version of mlir-xten currently fails to build against head torch-mlir. Fixes in progress.

### Compile torch-mlir

Clone torch-mlir from https://github.com/llvm/torch-mlir and follow the build instructions.  This process builds torch-mlir integrated with LLVM/MLIR.  We recommend adding the additional CMake flags: `-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON` to build and link with libLLVM.so.

```sh
cmake --build build
```

### Compile mlir-xten

```sh
git clone https://github.com/Xilinx/mlir-xten.git
mkdir build; cd build
cmake ..\
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${absolute path to torchMlirBuild}/cmake/llvm \
    -DMLIR_DIR=${absolute path to torchMlirBuild}/cmake/mlir \
    -DTORCH_MLIR_SOURCE_DIR=${absolute path to torchMlirSource} \
    -DTORCH_MLIR_BINARY_DIR=${absolute path to torchMlirBuild}
```

## Environment setup

TBD

-----

<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
