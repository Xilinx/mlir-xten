#ifndef XTEN_CONVERSION_PASSES_H
#define XTEN_CONVERSION_PASSES_H

#include "xten/Conversion/ATenToXTenPass.h"
#include "xten/Conversion/XTenToAffinePass.h"
#include "xten/Conversion/XTenToLinalgPass.h"

namespace xilinx {
namespace xten {

void registerConversionPasses();

} // namespace xten
} // namespace xilinx

#endif // XTEN_CONVERSION_PASSES_H