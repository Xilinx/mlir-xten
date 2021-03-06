//===- xten_conv2d_nobias.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -xten-to-affine -cse | FileCheck %s --check-prefix=LOOP
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// LOOP-LABEL:   func @conv2d(
// LOOP-SAME:                 %[[VAL_0:.*]]: tensor<16x16x1x1xf32>,
// LOOP-SAME:                 %[[VAL_1:.*]]: tensor<16x16x64x64xf32>) -> tensor<16x16x64x64xf32> {
// LOOP:           %[[VAL_2:.*]] = constant false
// LOOP:           %[[VAL_3:.*]] = constant 0 : i64
// LOOP:           %[[VAL_4:.*]] = constant 1 : i64
// LOOP:           %[[VAL_5:.*]] = basicpy.singleton : !basicpy.NoneType
// LOOP:           %[[VAL_6:.*]] = basicpy.build_list %[[VAL_4]], %[[VAL_4]] : (i64, i64) -> !basicpy.ListType
// LOOP:           %[[VAL_7:.*]] = basicpy.build_list %[[VAL_3]], %[[VAL_3]] : (i64, i64) -> !basicpy.ListType
// LOOP:           %[[VAL_8:.*]] = memref.alloc() : memref<16x16x64x64xf32>
// LOOP:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<16x16x64x64xf32>
// LOOP:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_0]] : memref<16x16x1x1xf32>
// LOOP:           affine.for %[[VAL_11:.*]] = 0 to 16 {
module  {
  func @conv2d(%arg0: tensor<16x16x1x1xf32>, %arg1: tensor<16x16x64x64xf32>) -> tensor<16x16x64x64xf32> {
    %false = constant false
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %0 = basicpy.singleton : !basicpy.NoneType
    %1 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
    %2 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
    %3 = "xten.conv2d"(%arg1, %arg0, %0, %1, %2, %1, %false, %2, %c1_i64) : (tensor<16x16x64x64xf32>, tensor<16x16x1x1xf32>, !basicpy.NoneType, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64) -> tensor<16x16x64x64xf32>
    return %3 : tensor<16x16x64x64xf32>
  }
}