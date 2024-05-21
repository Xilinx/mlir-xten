//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "xten/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "xten/Transform/Passes.h.inc"
}

void xilinx::xten::registerTransformPasses() { ::registerXTenTransformPasses(); }
