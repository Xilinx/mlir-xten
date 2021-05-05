// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#pragma once

#include "AirDataflowUtils.h"

#include <memory>

namespace mlir {
    class Pass;
}

namespace xilinx {
    namespace air {

        class DataflowExplorer {
        private:
            std::map<std::string, std::vector<Operation*>> layerNameToOps;
            std::map<std::string, std::vector<ModelParams>> validTopologies;

        public:
            // Initialized with un-expanded network topology
            DataflowExplorer(std::map<std::string, std::vector<Operation*>> &nameToOps);

            // Builds a decision tree and finds the best possible solution
            // Then fills the validTopologies structure
            // If called more than once do not recomputes anything
            ModelParams explore();

            // Supposedly according to some metric
            std::map<std::string, ModelParams> getBestTopology();
        };
    }
}

