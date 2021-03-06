//===- XTenDataflowExplorer.h -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "xten/Dialect/XTen/XTenDataflowUtils.h"
#include "xten/Dialect/XTen/XTenOpWrapper.h"
#include "xten/Util/Arch.h"

#define FORCE_INT8 1

#include <memory>
#include <math.h>

namespace mlir {
    class Pass;
}

namespace xilinx {
    namespace xten {
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
            uint64_t totalTime;

            Node_t(ModelParams p, uint64_t totalT) {
                params = p;
                totalTime = totalT;
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

            // Plotting all paths
            std::map<uint64_t, std::vector<bool>> perfToArea;

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
            double getUtilization(std::vector<ModelParams> &Params, unsigned int numCores);
            uint64_t getArea(std::vector<ModelParams> &params);

            bool allWeightsIn(uint64_t layerId, ModelParams &params);
            bool checkPath(std::vector<ModelParams> &params);

            // Explore functions
            bool isValid(uint64_t layerId, ModelParams &params);
            bool prune(Node_t* locNode, uint64_t locLayerId, Node_t* prevNode, uint64_t locBound, uint64_t prevBound, float maxDist);
            bool wMatches(Node_t* layerNode, Node_t* inNode, uint64_t layerId);
            std::vector<uint64_t> generateExplorationBounds();

            void generateValidTopologies();
            void generatePathGraph();
            void enumeratePaths();
            void getParetoFrontierAndCleanGraph();
            void dfsRec(Node_t* node, std::vector<ModelParams> path, uint64_t loc,
                        std::ofstream &throughput, std::ofstream &latency);
            void dfsRecFast(Node_t* node, std::vector<ModelParams> path, uint64_t loc,
                        std::ofstream &throughput, std::ofstream &latency);
            void dfs(bool full);

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
            void dumpMacs();

            std::map<std::string, ModelParams> getMaxThroughput();
            std::map<std::string, ModelParams> getBestTopology();
        };
    }
}

