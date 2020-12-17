#ifndef AIRPASSES_H_
#define AIRPASSES_H_

#include "mlir/Pass/Pass.h"
#include "AIRDialect.h"

#include "AIRLoweringPass.h"
#include "AIRToAffinePass.h"
#include "AffineToAIRPass.h"

namespace xilinx {
  namespace air {
// #define GEN_PASS_CLASSES
// #include "ATRPasses.h.inc"

    void registerAIRPasses();
  }
}
#endif // AIRPASSES_H_
