// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#pragma once

#include "AirDataflowUtils.h"
#include "AirOpWrapper.h"

#include <memory>
#include <math.h>

namespace mlir {
    class Pass;
}

namespace xilinx {
    namespace air {

        class AbsArchitecture {
        public:
            virtual ~AbsArchitecture() {};
            virtual uint64_t getBankSize() = 0;
            virtual uint64_t getNumBanks() = 0;
            virtual uint64_t getMemSize() = 0;
            virtual uint64_t getVectSize() = 0;
            virtual uint64_t getComSpeed() = 0;
            virtual uint64_t getPipelineDepth() = 0;
            virtual uint64_t getNumCores() = 0;
            virtual uint64_t getClockFrequency() = 0;
        };

        class AIEv1 : public AbsArchitecture {
        private:
            uint64_t xWidth;
            uint64_t zWidth;

        public:
            AIEv1(uint64_t acts, uint64_t weights) : xWidth(acts), zWidth(weights) {}
            ~AIEv1() {}

            // Size in bytes
            uint64_t getBankSize() {
                return pow(2, 12);
            }

            // Integer
            uint64_t getNumBanks() {
                return 8;
            }

            // Size in bytes
            uint64_t getMemSize() {
                return getBankSize() * getNumBanks();
            }

            // Integer
            uint64_t getVectSize() {
                return 128 / (xWidth * zWidth);
            }

            // Bytes per cycles
            uint64_t getComSpeed() {
                return 4;
            }

            // Integer, TODO check that
            uint64_t getPipelineDepth() {
                return 8;
            }

            uint64_t getNumCores() {
                return 400;
            }

            uint64_t getClockFrequency() {
                return pow(10, 9);
            }
        };

        class DataflowExplorer {
        private:
            std::vector<AbsOpWrapper*> layerNameToOps;
            std::map<std::string, uint64_t> layerNameToID;
            std::vector<std::vector<ModelParams>> validTopologies;
            AbsArchitecture* arch;

            // Analytical model functions
            uint64_t getLinesPerTile(uint64_t layerId, ModelParams &params);
            uint64_t getBanksPerLine(uint64_t layerId, ModelParams &params);
            uint64_t getK(uint64_t layerId, ModelParams &params);
            uint64_t getMissmatchChannels(int64_t dim, uint64_t params);
            uint64_t getMissmatchLines(int64_t dim, uint64_t params);

            uint64_t getComputeTimePerTile(uint64_t layerId, ModelParams &params);
            uint64_t getComputeTime(uint64_t layerId, ModelParams &params);

            uint64_t getActivationInBanks(uint64_t layerId, ModelParams &params);
            uint64_t getActivationOutBanks(uint64_t layerId, ModelParams &params);
            uint64_t getWeightBanks(uint64_t layerId, ModelParams &params);
            uint64_t getTotalMemBanks(uint64_t layerId, ModelParams &params);

            uint64_t getActCommunicationTimePerTile(uint64_t layerId, ModelParams &params);
            uint64_t getActCommunicationTime(uint64_t layerId, ModelParams &params);

            uint64_t getWeightCommunicationTimePerTile(uint64_t layerId, ModelParams &params);
            uint64_t getWeightCommunicationTime(uint64_t layerid, ModelParams &params);

            uint64_t getTotalTimePerTile(uint64_t layerId, ModelParams &params);
            uint64_t getTotalTime(uint64_t layerId, ModelParams &params);

            uint64_t getTotalCompute();

            uint64_t getEndToEndLatency(std::vector<ModelParams> &params);
            uint64_t getThroughput(std::vector<ModelParams> &params);
            double getUtilization(std::vector<ModelParams> &Params);
            uint64_t getArea(std::vector<ModelParams> &params);

            // Explore functions
            bool isValid(uint64_t layerId, ModelParams &params);
            std::vector<uint64_t> generateExplorationBounds();
        public:
            DataflowExplorer(std::vector<std::pair<std::string, AbsOpWrapper*>> &nameToOps);
            ~DataflowExplorer();

            // Explore function
            void generateValidTopologies();
            void printValidTopologies();
            void dumpValidTopologies();
            std::map<std::string, ModelParams> getBestTopology();


        };
    }
}

