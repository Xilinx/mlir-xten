// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>

namespace mlir {
    class Pass;
} // namespace mlir

namespace xilinx {
    namespace air {

        std::unique_ptr<mlir::Pass> createAirNamePass();
        void registerAirNamePass();

    } // namespace aten
} // namespace xilinx
