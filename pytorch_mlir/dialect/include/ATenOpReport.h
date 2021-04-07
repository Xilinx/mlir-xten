// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenOpReportPass(std::string &output);
void registerATenOpReportPass();

std::map<std::string, uint64_t> getATenOpStats(mlir::Operation *op);

} // namespace aten
} // namespace xilinx
