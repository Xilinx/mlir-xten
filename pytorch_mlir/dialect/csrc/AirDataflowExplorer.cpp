#include "AirDataflowExplorer.h"

#define DEBUG_TYPE "air-dataflow-explorer"

namespace xilinx {
    namespace air {
        DataflowExplorer::DataflowExplorer(std::map<std::string, std::vector<Operation*>> &nameToOps) {
            this->layerNameToOps = nameToOps;
        }

        ModelParams DataflowExplorer::explore() {
            // TODO
            return ModelParams();
        }
    }
}
