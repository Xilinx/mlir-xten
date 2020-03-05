// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#pragma once

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createReturnEliminationPass();

} // namespace aten
} // namespace xilinx
