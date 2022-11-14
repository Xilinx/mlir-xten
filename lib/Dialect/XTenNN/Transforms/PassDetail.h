//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Private include for pass implementations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "xten/Dialect/XTenNN/Transforms/XTenNNPasses.h"

namespace amd::xten_nn {
#define GEN_PASS_CLASSES
#include "xten/Dialect/XTenNN/Transforms/XTenNNPasses.h.inc"

} // namespace amd::xten_nn