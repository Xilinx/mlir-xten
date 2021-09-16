// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "aten/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aten/Transform/Passes.h.inc"
}

void xilinx::aten::registerTransformPasses() { ::registerATenTransformPasses(); }
