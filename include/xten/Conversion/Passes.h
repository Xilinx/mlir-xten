//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_CONVERSION_PASSES_H
#define XTEN_CONVERSION_PASSES_H

#include "xten/Conversion/ATenToXTenPass.h"
#include "xten/Conversion/TosaToXTenNNPass.h"
#include "xten/Conversion/XTenNNToTosaPass.h"
#include "xten/Conversion/XTenToAffinePass.h"
#include "xten/Conversion/XTenToLinalgPass.h"

namespace xilinx {
namespace xten {

void registerConversionPasses();

} // namespace xten
} // namespace xilinx

#endif // XTEN_CONVERSION_PASSES_H