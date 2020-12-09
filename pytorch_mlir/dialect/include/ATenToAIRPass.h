// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef ATEN_TO_AIR_H_
#define ATEN_TO_AIR_H_

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenToAIRPass();
  void registerATenToAIRPass();
} // namespace aten
} // namespace xilinx

#endif // ATEN_TO_AIR_H_
