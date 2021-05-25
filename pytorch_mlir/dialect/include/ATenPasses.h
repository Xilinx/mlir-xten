#ifndef ATENPASSES_H_
#define ATENPASSES_H_

#include "mlir/Pass/Pass.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "AcapHerdAssignPass.h"
#include "AcapHerdLoweringPass.h"
#include "AcapLoopLoweringPass.h"
#include "AffineLoopOptPass.h"
#include "ATenToAIRPass.h"
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
#endif // ATENPASSES_H_
