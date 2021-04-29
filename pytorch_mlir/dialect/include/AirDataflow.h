// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>

namespace mlir {
    class Pass;
}

namespace xilinx {
    namespace air {

        std::unique_ptr<mlir::Pass> createAirDataflowPass();
        void registerAirDataflowPass();

    }
}

