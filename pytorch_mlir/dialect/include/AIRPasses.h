#ifndef AIRPASSES_H_
#define AIRPASSES_H_

#include "mlir/Pass/Pass.h"
#include "AIRDialect.h"

#include "AIRLoweringPass.h"
#include "AIRToAffinePass.h"
#include "AffineToAIRPass.h"
#include "AIRToAIEPass.h"
#include "AIRToLinalgPass.h"
#include "AIRLinalgCodegen.h"
#include "XTenDataflow.h"
#include "XTenNamePass.h"
#include "AIRAutomaticTilingPass.h"
#include "AIRRegularizeLoopPass.h"
#include "AIRLoopPermutationPass.h"

namespace xilinx {
  namespace air {
// #define GEN_PASS_CLASSES
// #include "AIRPasses.h.inc"

    void registerAIRPasses();
  }
}
#endif // AIRPASSES_H_
