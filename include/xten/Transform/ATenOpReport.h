// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifndef ATEN_OP_REPORT_H
#define ATEN_OP_REPORT_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::Pass> createATenOpReportPass(std::string &output);
std::unique_ptr<mlir::Pass> createATenOpReportPass();

std::map<std::string, uint64_t> getATenOpStats(mlir::Operation *op);

} // namespace xten
} // namespace xilinx

#endif // ATEN_OP_REPORT_H