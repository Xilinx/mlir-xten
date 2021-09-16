#ifndef ATEN_CONVERSION_PASSDETAIL_H
#define ATEN_CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace aten {

using namespace mlir;

#define GEN_PASS_CLASSES
#include "aten/Conversion/Passes.h.inc"

} // namespace aten
} // namespace xilinx

#endif // ATEN_CONVERSION_PASSDETAIL_H