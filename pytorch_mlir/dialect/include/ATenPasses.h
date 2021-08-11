#ifndef ATEN_PASSES_H
#define ATEN_PASSES_H

#include "mlir/Pass/Pass.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "AcapHerdAssignPass.h"
#include "AcapHerdLoweringPass.h"
#include "AcapLoopLoweringPass.h"
#include "ATenLoweringPass.h"
#include "ATenSimpleAllocatePass.h"
#include "ATenOpReport.h"
#include "ReturnEliminationPass.h"
#include "LowerToLibATenPass.h"

namespace xilinx {
  namespace aten {
// #define GEN_PASS_CLASSES
// #include "ATenPasses.h.inc"

    void registerATenPasses();
  }
}
#endif // ATENPASSES_H
