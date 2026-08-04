<!--- (c) Copyright 2021 Xilinx, Inc. All Rights reserved.--->
<!--- (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.--->

# Building the code

## Prerequisites

```
cmake 3.20.6 or higher
ninja 1.8.2
clang/llvm 14+ from source https://github.com/llvm/llvm-project
```

## Building on X86

### Compile mlir-xten

```sh
git clone https://github.com/Xilinx/mlir-xten.git
mkdir build; cd build
cmake ..\
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${absolute path to llvm build}/cmake/llvm \
    -DMLIR_DIR=${absolute path to mlir build}/cmake/mlir
```

## Environment setup

TBD

-----

<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
