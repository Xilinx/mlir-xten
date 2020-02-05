// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace reports {

std::unique_ptr<mlir::Pass> createATenLayerNamePass();

} // namespace reports
} // namespace xilinx
