//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#ifndef XTEN_CONVERSION_PASSES_H
#define XTEN_CONVERSION_PASSES_H

#include "xten/Conversion/TosaToXTenNNPass.h"
#include "xten/Conversion/XTenNNToLinalgPass.h"
#include "xten/Conversion/XTenNNToTorchPass.h"
#include "xten/Conversion/XTenNNToTosaPass.h"

namespace xilinx {
namespace xten {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xten/Conversion/Passes.h.inc"

void registerConversionPasses();

} // namespace xten
} // namespace xilinx

#endif // XTEN_CONVERSION_PASSES_H
