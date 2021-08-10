// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "ATenPasses.h"

namespace xilinx {
namespace aten {
#define GEN_PASS_REGISTRATION
#include "ATenPasses.h.inc"
}
}

void xilinx::aten::registerATenPasses() {
// #define GEN_PASS_REGISTRATION
//   #include "ATenPasses.h.inc"
  xilinx::aten::registerAcapHerdAssignPass();
  xilinx::aten::registerAcapHerdLoweringPass();
  xilinx::aten::registerAcapLoopLoweringPass();
  xilinx::aten::registerAffineLoopOptPass();
  xilinx::aten::registerATenToXTenPass();
  xilinx::aten::registerATenOpReportPass();
  xilinx::aten::registerATenLoweringPass();
  xilinx::aten::registerATenSimpleAllocatePass();
  xilinx::aten::registerReturnEliminationPass();
  xilinx::aten::registerLowerToLibATenPass();
}
