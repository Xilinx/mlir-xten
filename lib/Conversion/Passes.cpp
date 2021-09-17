// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "xten/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "xten/Conversion/Passes.h.inc"
}

void xilinx::xten::registerConversionPasses() { ::registerXTenConversionPasses(); }
