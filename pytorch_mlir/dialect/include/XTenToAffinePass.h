// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>
#include <vector>

#include "ATenPasses.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace xten {

extern std::vector<uint64_t> Conv2dLoopOrder;
extern std::vector<uint64_t> Conv2dCopyDepth;
extern std::vector<uint64_t> Conv2dTileSizes;

std::unique_ptr<mlir::Pass> createXTenToAffinePass();
void registerXTenToAffinePass();

} // namespace xten
} // namespace xilinx
