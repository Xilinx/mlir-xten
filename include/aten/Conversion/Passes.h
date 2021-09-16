#ifndef ATEN_CONVERSION_PASSES_H
#define ATEN_CONVERSION_PASSES_H

#include "aten/Conversion/ATenToXTenPass.h"
#include "aten/Conversion/XTenToAffinePass.h"
#include "aten/Conversion/XTenToLinalgPass.h"

namespace xilinx {
namespace aten {

void registerConversionPasses();

} // namespace aten
} // namespace xilinx

#endif // ATEN_CONVERSION_PASSES_H