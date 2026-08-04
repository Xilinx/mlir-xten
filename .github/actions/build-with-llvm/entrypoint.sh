#!/bin/sh -l

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

echo "Hello $1"
time=$(date)
echo "::set-output name=time::$time"
cd ${GITHUB_WORKSPACE}
ls
# temporary, should be in prereq image

#apt-get install python3-dev
#pip3 install pybind11


#pybind11_DIR=/build/.pyenv/versions/3.7.0/lib/python3.7/site-packages/pybind11/share/cmake/pybind11
#pybind11_DIR=/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11
cmake -B build -S . -G Ninja \
    -DMLIR_DIR=/build/build/lib/cmake/mlir

#https://github.com/Xilinx/mlir-xten.git
#    -Dpybind11_DIR=${pybind11_DIR}

cmake --build build --target install -- -j$(nproc)