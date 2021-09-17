#ifndef XTEN_TRANSFORM_PASSES_H
#define XTEN_TRANSFORM_PASSES_H

#include "xten/Transform/ATenOpReport.h"
#include "xten/Transform/ATenLoweringPass.h"
#include "xten/Transform/LowerToLibATenPass.h"

namespace xilinx {
namespace xten {

void registerTransformPasses();

} // namespace xten
} // namespace xilinx

#endif // XTEN_TRANSFORM_PASSES_H