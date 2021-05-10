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
            virtual unsigned int getBankSize() = 0;
            virtual unsigned int getNumBanks() = 0;
            virtual unsigned int getMemSize() = 0;
            virtual unsigned int getVectSize() = 0;
            virtual unsigned int getComSpeed() = 0;
            virtual unsigned int getPipelineDepth() = 0;
            virtual unsigned int getNumCores() = 0;
        };

        class AIEv1 : public AbsArchitecture {
        private:
            unsigned int xWidth;
            unsigned int zWidth;

        public:
            AIEv1(unsigned int acts, unsigned int weights) : xWidth(acts), zWidth(weights) {}
            ~AIEv1() {}

            // Size in bytes
            unsigned int getBankSize() {
                return pow(2, 12);
            }

            // Integer
            unsigned int getNumBanks() {
                return 8;
            }

            // Size in bytes
            unsigned int getMemSize() {
                return getBankSize() * getNumBanks();
            }

            // Integer
            unsigned int getVectSize() {
                return 128 / (xWidth * zWidth);
            }

            // Bytes per cycles
            unsigned int getComSpeed() {
                return 4;
            }

            // Integer, TODO check that
            unsigned int getPipelineDepth() {
                return 8;
            }

            unsigned int getNumCores() {
                return 400;
            }
        };

        class DataflowExplorer {
        private:
            std::vector<AbsOpWrapper*> layerNameToOps;
            std::map<std::string, uint64_t> layerNameToID;
            std::vector<std::vector<ModelParams>> validTopologies;
            AbsArchitecture* arch;

            // Analytical model functions
            unsigned int getLinesPerTile(unsigned int layerId, ModelParams &params);
            unsigned int getBanksPerLine(unsigned int layerId, ModelParams &params);
            unsigned int getK(unsigned int layerId, ModelParams &params);
            unsigned int getMissmatchChannels(int64_t dim, unsigned int params);
            unsigned int getMissmatchLines(int64_t dim, unsigned int params);

            unsigned int getComputeTimePerTile(unsigned int layerId, ModelParams &params);
            unsigned int getComputeTime(unsigned int layerId, ModelParams &params);

            unsigned int getActivationInBanks(unsigned int layerId, ModelParams &params);
            unsigned int getActivationOutBanks(unsigned int layerId, ModelParams &params);
            unsigned int getWeightBanks(unsigned int layerId, ModelParams &params);
            unsigned int getTotalMemBanks(unsigned int layerId, ModelParams &params);

            unsigned int getActCommunicationTimePerTile(unsigned int layerId, ModelParams &params);
            unsigned int getActCommunicationTime(unsigned int layerId, ModelParams &params);

            unsigned int getWeightCommunicationTimePerTile(unsigned int layerId, ModelParams &params);
            unsigned int getWeightCommunicationTime(unsigned int layerid, ModelParams &params);

            unsigned int getTotalTimePerTile(unsigned int layerId, ModelParams &params);
            unsigned int getTotalTime(unsigned int layerId, ModelParams &params);

            // Explore functions
            bool isValid(unsigned int layerId, ModelParams &params);
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

