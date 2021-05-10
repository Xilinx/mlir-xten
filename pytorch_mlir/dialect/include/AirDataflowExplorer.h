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
            virtual ~AbsArchitecture() = 0;
            virtual unsigned int getBankSize();
            virtual unsigned int getNumBanks();
            virtual unsigned int getMemSize();
            virtual unsigned int getVectSize();
            virtual unsigned int getComSpeed();
            virtual unsigned int getPipelineDepth();
        };

        class AIEv1 : public AbsArchitecture {
        private:
            unsigned int xWidth;
            unsigned int zWidth;

        public:
            AIEv1(unsigned int acts, unsigned int weights) : xWidth(acts), zWidth(weights) {}

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
        };

        class DataflowExplorer {
        private:
            std::map<std::string, AbsOpWrapper*> layerNameToOps;
            std::map<std::string, std::vector<ModelParams>> validTopologies;
            AbsArchitecture* arch;

            // Analytical model functions
            unsigned int getLinesPerTile(std::string layer, ModelParams &params);
            unsigned int getBanksPerLine(std::string layer, ModelParams &params);
            unsigned int getK(std::string layerName, ModelParams &params);
            unsigned int getMissmatchChannels(int64_t dim, unsigned int params);
            unsigned int getMissmatchLines(int64_t dim, unsigned int params);

            unsigned int getComputeTimePerTile(std::string layerName, ModelParams &params);
            unsigned int getComputeTime(std::string layer, ModelParams &params);

            unsigned int getActivationInBanks(std::string layer, ModelParams &params);
            unsigned int getActivationOutBanks(std::string layer, ModelParams &params);
            unsigned int getWeightBanks(std::string layer, ModelParams &params);
            unsigned int getTotalMemBanks(std::string layer, ModelParams &params);

            unsigned int getActCommunicationTimePerTile(std::string layer, ModelParams &params);
            unsigned int getActCommunicationTime(std::string layer, ModelParams &params);

            unsigned int getWeightCommunicationTimePerTile(std::string layer, ModelParams &params);
            unsigned int getWeightCommunicationTime(std::string, ModelParams &params);

            unsigned int getTotalTimePerTile(std::string layerName, ModelParams &params);
            unsigned int getTotalTime(std::string layer, ModelParams &params);
        public:
            DataflowExplorer(std::map<std::string, AbsOpWrapper*> &nameToOps);
            ~DataflowExplorer();

            // Explore function
            void generateValidTopologies();
            std::map<std::string, ModelParams> getBestTopology();


        };
    }
}

