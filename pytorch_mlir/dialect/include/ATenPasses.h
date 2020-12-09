#ifndef ATENPASSES_H_
#define ATENPASSES_H_

#include "mlir/Pass/Pass.h"
#include "ATenDialect.h"

#include "AcapHerdAssignPass.h"
#include "AcapHerdLoweringPass.h"
#include "AcapLoopLoweringPass.h"
#include "AffineLoopOptPass.h"
#include "ATenToAIRPass.h"
#include "ATenLayerNamePass.h"
#include "ATenLoweringPass.h"
#include "ATenSimpleAllocatePass.h"
#include "ReturnEliminationPass.h"

namespace xilinx {
  namespace aten {
// #define GEN_PASS_CLASSES
// #include "ATenPasses.h.inc"

    void registerATenPasses();
  }
}
#endif // ATENPASSES_H_
