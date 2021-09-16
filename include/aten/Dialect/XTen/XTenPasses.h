#ifndef XTENPASSES_H_
#define XTENPASSES_H_

#include "mlir/Pass/Pass.h"

#include "aten/Dialect/XTen/XTenDataflow.h"
//#include "aten/Dialect/XTen/XTenToAffinePass.h"
#include "aten/Dialect/XTen/XTenNamePass.h"

namespace xilinx {
  namespace xten {
// #define GEN_PASS_CLASSES
// #include "XTenPasses.h.inc"

    void registerXTenPasses();
  }
}
#endif // XTENPASSES_H_
