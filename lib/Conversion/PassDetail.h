#ifndef XTEN_CONVERSION_PASSDETAIL_H
#define XTEN_CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

using namespace mlir;

#define GEN_PASS_CLASSES
#include "xten/Conversion/Passes.h.inc"

} // namespace xten
} // namespace xilinx

#endif // XTEN_CONVERSION_PASSDETAIL_H