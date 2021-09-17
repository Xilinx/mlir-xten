// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>

namespace mlir {
    class Pass;
} // namespace mlir

namespace xilinx {
    namespace xten {

        std::unique_ptr<mlir::Pass> createXTenNamePass();
        void registerXTenNamePass();

    } // namespace xten
} // namespace xilinx
