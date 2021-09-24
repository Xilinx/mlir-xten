// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "xten/Dialect/XTen/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "xten/Dialect/XTen/Passes.h.inc"
}

namespace xilinx {
namespace xten {

void registerXTenPasses() {
  ::registerXTenDataFlowPass();
}

}
}