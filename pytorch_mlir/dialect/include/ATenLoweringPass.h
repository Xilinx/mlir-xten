// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef ATEN_LOWERING_H_
#define ATEN_LOWERING_H_

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace xilinx {
  namespace aten {
    std::unique_ptr<mlir::Pass> createATenLoweringPass();
  }
} // namespace xilinx::aten

#endif // ATEN_LOWERING_H_
