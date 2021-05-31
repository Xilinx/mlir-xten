// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#pragma once

#include "AirDataflowUtils.h"
#include "AirOpWrapper.h"
#include "Arch.h"

#define FORCE_INT8 1

#include <memory>
#include <math.h>

namespace mlir {
    class Pass;
}

namespace xilinx {
    namespace air {
        static unsigned int getElementWidth(ShapedType tensorType, bool forceINT8) {
            if(forceINT8) {
                return 1;
            } else {
                return (tensorType.getElementTypeBitWidth() / 8);
            }
        }

        class PathInfo_t {
        public:
            std::vector<ModelParams> path;
            uint64_t maxTotalTime;
            uint64_t value; // Either throughput or latency or quantity of interest

            PathInfo_t(uint64_t startValue) {
                maxTotalTime = 0;
                value = startValue;
            }
        };

        // TODO investigate if it's to big to keep model params here or no
        class Node_t {
        public:
            ModelParams params;
            std::vector<Node_t*> ins;

            // Maps an area to a path
            // area is in # of cores and is the index
            std::vector<PathInfo_t> areaToThroughput;
            std::vector<PathInfo_t> areaToLatency;

            Node_t(ModelParams p) {
                params = p;
            }
        };

        // TODO build destructors for graphs

        class DataflowExplorer {
        public:
            std::vector<AbsOpWrapper*> layerNameToOps;
            std::vector<std::map<std::string, int64_t>> layerNameToSize;
            std::map<std::string, uint64_t> layerNameToID;
            std::map<uint64_t, std::string> layerIdToName;
            std::vector<std::vector<ModelParams>> validTopologies;
            std::vector<std::vector<Node_t*>> pathGraph;
            AbsArchitecture* arch;

            // Analytical model functions
            uint64_t getLinesPerTile(uint64_t layerId, ModelParams &params);
            uint64_t getBanksPerLine(uint64_t layerId, ModelParams &params);
            uint64_t getK(uint64_t layerId, ModelParams &params);
            uint64_t getMissmatchChannels(int64_t dim, uint64_t params);
            uint64_t getMissmatchLines(int64_t dim, uint64_t params);
            uint64_t getTilesPerCore(uint64_t layerId, ModelParams &params);

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

            double getLayerUtilization(uint64_t layerId, ModelParams &params);

            uint64_t getTotalCompute();
            std::vector<uint64_t> getMemWeightPerLayer();

            uint64_t getEndToEndLatency(std::vector<ModelParams> &params);
            uint64_t getThroughput(std::vector<ModelParams> &params);
            double getUtilization(std::vector<ModelParams> &Params);
            uint64_t getArea(std::vector<ModelParams> &params);

            // Explore functions
            bool isValid(uint64_t layerId, ModelParams &params);
            bool wMatches(Node_t* layerNode, Node_t* inNode, uint64_t layerId);
            std::vector<uint64_t> generateExplorationBounds();

            void generateValidTopologies();
            void generatePathGraph();
            void enumeratePaths();
            void getParetoFrontierAndCleanGraph();

            // Pareto stuff found at the end of exploration
            std::vector<PathInfo_t> paretoThroughput;
            std::vector<PathInfo_t> paretoLatency;

            DataflowExplorer(std::vector<std::pair<std::string, AbsOpWrapper*>> &nameToOps);
            ~DataflowExplorer();

            // Explore function
            void enumerate();
            void printValidTopologies();
            void dumpModelParam(ModelParams& params, std::ofstream &outputFile, std::string layerName, uint64_t i);
            void dumpValidTopologies();
            void dumpParetoFrontiers();
            void dumpPath(PathInfo_t &path, std::string fname);
            void dumpPathsFrom(std::vector<PathInfo_t> &paths, std::string prefix);

            std::map<std::string, ModelParams> getMaxThroughput();
            std::map<std::string, ModelParams> getBestTopology();
        };
    }
}

