// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>
#include <vector>

#include "ATenPasses.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createLowerToLibATenPass();
void registerLowerToLibATenPass();

} // namespace aten
} // namespace xilinx
