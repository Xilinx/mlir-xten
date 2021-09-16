#ifndef ATEN_TRANSFORM_PASSES_H
#define ATEN_TRANSFORM_PASSES_H

#include "aten/Transform/ATenOpReport.h"
#include "aten/Transform/ATenLoweringPass.h"
#include "aten/Transform/LowerToLibATenPass.h"

namespace xilinx {
namespace aten {

void registerTransformPasses();

} // namespace aten
} // namespace xilinx

#endif // ATEN_TRANSFORM_PASSES_H