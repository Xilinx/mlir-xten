// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "xten/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "xten/Transform/Passes.h.inc"
}

void xilinx::xten::registerTransformPasses() { ::registerXTenTransformPasses(); }
