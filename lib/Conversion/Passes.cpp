// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "aten/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aten/Conversion/Passes.h.inc"
}

void xilinx::aten::registerConversionPasses() { ::registerATenConversionPasses(); }
