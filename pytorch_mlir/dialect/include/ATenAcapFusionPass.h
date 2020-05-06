// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef ATEN_FUSION_H_
#define ATEN_FUSION_H_

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createATenAcapFusionPass();

} // namespace aten
} // namespace xilinx

#endif // ATEN_FUSION_H_
