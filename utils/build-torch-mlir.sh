#!/usr/bin/env bash
##===- utils/build-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with the standard options. Intended to be called from 
# the github workflows.
#
##===----------------------------------------------------------------------===##

BUILD_DIR=${1:-"torch-mlir-build"}
INSTALL_DIR=${2:-"torch-mlir-install"}
SOURCE_DIR="torch-mlir"

export commithash=52ed3313b49b8f56a6dac270ee334c6dffebcaf2

echo "dirs:$BUILD_DIR $INSTALL_DIR"
git clone --depth 10000 https://github.com/llvm/torch-mlir.git ${SOURCE_DIR}
pushd ${SOURCE_DIR}
git checkout $commithash
popd

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cmake -B $BUILD_DIR -S $SOURCE_DIR -G Ninja \
    -DMLIR_DIR=`pwd`/build/lib/cmake/mlir/ \
    -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
    -DCMAKE_C_COMPILER=clang-8 \
    -DCMAKE_CXX_COMPILER=clang++-8 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build $BUILD_DIR --target install -- -j$(nproc)
