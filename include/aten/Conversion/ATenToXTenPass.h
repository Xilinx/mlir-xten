// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef ATEN_TO_XTEN_H
#define ATEN_TO_XTEN_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createATenToXTenPass();

}  // namespace xten
} // namespace xilinx

#endif // ATEN_TO_XTEN_H