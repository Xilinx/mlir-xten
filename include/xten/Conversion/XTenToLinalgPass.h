// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef XTEN_TO_LINALG_H
#define XTEN_TO_LINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createXTenToLinalgPass();

} // namespace xten
} // namespace xilinx

#endif // XTEN_TO_LINALG_H