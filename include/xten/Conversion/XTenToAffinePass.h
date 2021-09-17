// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef ATEN_TO_AFFINE_PASS_H
#define ATEN_TO_AFFINE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace xten {

extern std::vector<uint64_t> Conv2dLoopOrder;
extern std::vector<uint64_t> Conv2dCopyDepth;
extern std::vector<uint64_t> Conv2dTileSizes;

std::unique_ptr<mlir::Pass> createXTenToAffinePass();

} // namespace xten
} // namespace xilinx

#endif // ATEN_TO_AFFINE_PASS_H