#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "AirDataflowExplorer.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"

#include "ATenOpReport.h"

#include <iostream>
#include <fstream>

#define DEBUG_TYPE "air-dataflow-explorer"

#define MARGIN 2

// TODO also take into account kernel efficiency?

namespace xilinx {
    namespace air {
        // TODO for now both types are the same elementType
        DataflowExplorer::DataflowExplorer(std::vector<std::pair<std::string, AbsOpWrapper*>> &nameToOps) {
            uint64_t id = 0;
            for(auto pair : nameToOps) {
                this->layerNameToID[pair.first] = id;
                this->layerNameToOps.push_back(pair.second);
                id++;
            }

            this->validTopologies = std::vector<std::vector<ModelParams>>(id, std::vector<ModelParams>());
            uint64_t aWidth = this->layerNameToOps[0]->getInput().getType().dyn_cast<ShapedType>().getElementTypeBitWidth() / 8;
            this->arch = new AIEv1(aWidth, aWidth);
        }

        DataflowExplorer::~DataflowExplorer() {
            delete this->arch;
        }

        // Analytical model functions

        uint64_t DataflowExplorer::getLinesPerTile(uint64_t layerId, ModelParams &params) {
            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t linesPerBanks = (int64_t)floor(this->arch->getBankSize() / (float)lineSize);
            return linesPerBanks;
        }

        uint64_t DataflowExplorer::getBanksPerLine(uint64_t layerId, ModelParams& params) {
            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t banksPerLine = (int64_t)ceil((float)lineSize / this->arch->getBankSize());
            return banksPerLine;
        }

        // Analytical model functions
        uint64_t DataflowExplorer::getActivationInBanks(uint64_t layerId, ModelParams &params) {
            uint64_t F0 = this->layerNameToOps[layerId]->getKernelSize();
            int64_t linesPerBanks = this->getLinesPerTile(layerId, params);
            if(linesPerBanks != 0) {
                return ceil(ceil((float)F0 / params.L) / linesPerBanks);
            } else {
                uint64_t banksPerLine = this->getBanksPerLine(layerId, params);
                return ceil((float)F0 / params.L) * banksPerLine;
            }
        }

        uint64_t DataflowExplorer::getActivationOutBanks(uint64_t layerId, ModelParams &params) {
            return 2; // simple 4KB PingPong buffers
        }

        // either 2 or 4
        uint64_t DataflowExplorer::getWeightBanks(uint64_t layerId, ModelParams &params) {
            if(!this->layerNameToOps[layerId]->hasWeights()) {
                return 0;
            }

            ShapedType wShape = this->layerNameToOps[layerId]->getWeights().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> wShapeAR = wShape.getShape();
            int64_t COut = wShapeAR[COUT_LOC];
            int64_t CIn = wShapeAR[CIN_LOC];
            int64_t F0 = wShapeAR[F0_LOC];
            int64_t F1 = wShapeAR[F1_LOC];

            int64_t weightSize = COut * CIn * F0 * F1 * wShape.getElementTypeBitWidth() / 8;
            int64_t weightBanks = ceil(weightSize / this->arch->getBankSize());

            if((weightBanks > 4) || (weightBanks == 3)) {
                return 4;
            } else {
                return weightBanks;
            }
        }

        uint64_t DataflowExplorer::getTotalMemBanks(uint64_t layerId, ModelParams &params) {
            uint64_t inBanks = this->getActivationInBanks(layerId, params);
            uint64_t outBanks = this->getActivationOutBanks(layerId, params);
            uint64_t weightBanks = this->getWeightBanks(layerId, params);

            return inBanks + outBanks + weightBanks;
        }

        uint64_t DataflowExplorer::getMissmatchChannels(int64_t dim, uint64_t param) {
            uint64_t allGet = floor((float)dim / param) / 8;
            uint64_t someGet = dim / 8 - allGet * param;
            return someGet;
        }

        uint64_t DataflowExplorer::getMissmatchLines(int64_t dim, uint64_t param) {
            uint64_t allGet = floor((float)dim / param);
            uint64_t someGet = dim - allGet * param;
            return someGet;
        }

        uint64_t DataflowExplorer::getK(uint64_t layerId, ModelParams &params) {
            uint64_t linesPerTile = this->getLinesPerTile(layerId, params);

            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t N = aShapeAR[N_LOC];

            uint64_t K = N / linesPerTile;
            return K;
        }

        uint64_t DataflowExplorer::getComputeTimePerTile(uint64_t layerId, ModelParams &params) {
            uint64_t K = this->getK(layerId, params);
            return this->getComputeTime(layerId, params) / K;
        }

        std::map<std::string, uint64_t> getStats(Operation* op) {
            std::map<std::string, uint64_t> layerStatsMap;
            if (auto stats = llvm::dyn_cast<NPCOMP::StatisticsOpInterface>(op)) {
                layerStatsMap = stats.getStatistics();
            } else {
                layerStatsMap = xilinx::aten::getATenOpStats(op);
            }

            if(!layerStatsMap.size()) {
                llvm::outs() << "No statistics provided for that op!!\n";
                exit(1);
            }

            return layerStatsMap;
        }

        uint64_t DataflowExplorer::getComputeTime(uint64_t layerId, ModelParams &params) {
            std::map<std::string, uint64_t> stats = getStats(this->layerNameToOps[layerId]->getUnderlyingOperation());
            uint64_t macs = (stats.count("ops:MAC") == 0) ? stats["ops:>"] : stats["ops:MAC"];

            AbsOpWrapper* layer = this->layerNameToOps[layerId];
            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t CIn = aShapeAR[C_LOC];
            int64_t COut;
            if(layer->hasWeights()) {
                ShapedType wShape = layer->getWeights().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> wShapeAR = wShape.getShape();
                COut = wShapeAR[COUT_LOC];
            } else {
                COut = CIn;
            }

            int64_t F = layer->getKernelSize();

            uint64_t missmatchCa = getMissmatchChannels(CIn, params.Ca);
            uint64_t missmatchP = getMissmatchChannels(COut, params.P);
            uint64_t missmatchL = getMissmatchLines(F, params.L);

            // TODO double check this expression
            // TODO what about efficicency here?
            uint64_t time  = macs / ((params.P - missmatchP) * (params.Ca - missmatchCa) * (params.L - missmatchL) * params.W);
            return time / this->arch->getVectSize();
        }

        uint64_t DataflowExplorer::getActCommunicationTimePerTile(uint64_t layerId, ModelParams &params) {
            uint64_t K = this->getK(layerId, params);
            return this->getActCommunicationTime(layerId, params) / K;
        }

        uint64_t DataflowExplorer::getActCommunicationTime(uint64_t layerId, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerId];
            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            uint64_t C = aShapeAR[C_LOC];
            uint64_t M = aShapeAR[M_LOC];
            uint64_t N = aShapeAR[N_LOC];

            int64_t actSize = C * M * N * aShape.getElementTypeBitWidth() / 8;
            return actSize / this->arch->getComSpeed();
        }

        uint64_t DataflowExplorer::getWeightCommunicationTimePerTile(uint64_t layerId, ModelParams &params) {
            if(!this->layerNameToOps[layerId]->hasWeights()) {
                return 0;
            }

            ShapedType wShape = this->layerNameToOps[layerId]->getWeights().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> wShapeAR = wShape.getShape();
            int64_t COut = wShapeAR[COUT_LOC];
            int64_t CIn = wShapeAR[CIN_LOC];
            int64_t F0 = wShapeAR[F0_LOC];
            int64_t F1 = wShapeAR[F1_LOC];

            int64_t weightSize = COut * CIn * F0 * F1 * wShape.getElementTypeBitWidth() / 8;
            int64_t weightBanks = ceil(weightSize / this->arch->getBankSize());

            if(weightBanks <= 4) {
                return 0;
            }

            return weightSize / this->arch->getComSpeed();
        }

        uint64_t DataflowExplorer::getWeightCommunicationTime(uint64_t layerId, ModelParams &params) {
            uint64_t comPerTile = this->getWeightCommunicationTimePerTile(layerId, params);
            uint64_t K = this->getK(layerId, params);
            return comPerTile * K;
        }

        uint64_t DataflowExplorer::getTotalTimePerTile(uint64_t layerId, ModelParams &params) {
            uint64_t weightComTile = this->getWeightCommunicationTimePerTile(layerId, params);
            uint64_t actComTile = this->getActCommunicationTimePerTile(layerId, params);
            uint64_t computeTile = this->getComputeTimePerTile(layerId, params);

            // finds the bottleneck
            return std::max(std::max(actComTile, weightComTile), computeTile);
        }

        uint64_t DataflowExplorer::getTotalTime(uint64_t layerId, ModelParams &params) {
            uint64_t totalTimeTile = this->getTotalTimePerTile(layerId, params);
            uint64_t K = this->getK(layerId, params);

            return K * totalTimeTile;
        }

        uint64_t DataflowExplorer::getTotalCompute() {
            uint64_t totalCompute = 0;
            for(AbsOpWrapper* wrapped : this->layerNameToOps) {
                Operation* op = wrapped->getUnderlyingOperation();
                std::map<std::string, uint64_t> stats = getStats(op);
                uint64_t macs = (stats.count("ops:MAC") == 0) ? stats["ops:>"] : stats["ops:MAC"];
                totalCompute += macs;
            }

            return totalCompute;
        }

        // TODO at the moment for the next 4 functions assume start from layer 0 if not all layers are present
        // TODO might want to change that in the future
        uint64_t DataflowExplorer::getEndToEndLatency(std::vector<ModelParams> &params) {
            uint64_t latency = 0;
            for(uint64_t i = 0; i < params.size(); i++) {
                latency += this->getTotalTime(i, params.at(i));
            }

            return latency;
        }

        uint64_t DataflowExplorer::getThroughput(std::vector<ModelParams> &params) {
            uint64_t throughput = 0;
            for(uint64_t i = 0; i < params.size(); i++) {
                uint64_t totalTimeLayer = this->getTotalTime(i, params.at(i));
                uint64_t layerThroughput = (uint64_t)(1/(totalTimeLayer * pow(10, -9)));
                if(layerThroughput > throughput) {
                    throughput = layerThroughput;
                }
            }

            return throughput;
        }

        // Computes utilizaton of whole array
        double DataflowExplorer::getUtilization(std::vector<ModelParams> &params) {
            uint64_t maxWorkDone = this->arch->getNumCores() * this->arch->getClockFrequency();
            uint64_t throughput = this->getThroughput(params);
            uint64_t computePerSample = this->getTotalCompute();
            uint64_t workDone = throughput * computePerSample;
            return (double)workDone / maxWorkDone;
        }

        uint64_t DataflowExplorer::getArea(std::vector<ModelParams> &params) {
            uint64_t numCores = 0;
            for(ModelParams p : params) {
                numCores += p.cores();
            }

            return numCores;
        }

        // Explore functions
        std::vector<uint64_t>  DataflowExplorer::generateExplorationBounds() {
            uint64_t numCores = this->arch->getNumCores();
            std::vector<uint64_t> macsPerLayer;

            uint64_t sum = 0;
            for(AbsOpWrapper* elem : this->layerNameToOps) {
                std::map<std::string, uint64_t> stats = getStats(elem->getUnderlyingOperation());
                uint64_t macs = (stats.count("ops:MAC") == 0) ? stats["ops:>"] : stats["ops:MAC"];

                macsPerLayer.push_back(macs);
                sum += macs;
            }

            for(uint64_t i = 0; i < macsPerLayer.size(); i++) {
                macsPerLayer[i] = (macsPerLayer[i] * MARGIN / sum) * numCores;
            }

            //std::transform(macsPerLayer.begin(), macsPerLayer.end(), macsPerLayer.begin(),
            //               [&sum, &numCores](uint64_t macs) -> uint64_t {return (macs / sum) * MARGIN * numCores;});

            return macsPerLayer;
        }

        // Is not valid if does not fit under the memory constraints
        bool DataflowExplorer::isValid(uint64_t layerId, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerId];

            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t CIn = aShapeAR[C_LOC];
            int64_t COut;
            if(layer->hasWeights()) {
                ShapedType wShape = layer->getWeights().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> wShapeAR = wShape.getShape();
                COut = wShapeAR[COUT_LOC];
            } else {
                COut = CIn;
            }

            uint64_t F0 = layer->getKernelSize();

            bool memFit = this->getTotalMemBanks(layerId, params) < this->arch->getNumBanks();
            bool enoughCIn = (CIn / params.Ca) >= 8;
            bool enoughCOut = (COut / params.P) >= 8;
            bool enoughF = (F0 / params.L) >= 1;

            // TODO also check that W is rate balanced ?

            return memFit && enoughCIn && enoughCOut && enoughF;
        }

        void DataflowExplorer::generateValidTopologies() {
            std::vector<uint64_t> bounds = this->generateExplorationBounds();

            for(uint64_t layerId = 0; layerId < bounds.size(); layerId++) {
                uint64_t layerCores = bounds.at(layerId);
                uint64_t F0 = this->layerNameToOps[layerId]->getKernelSize();
                for(uint64_t p = 1; p < layerCores; p++) {
                    for(uint64_t ca = 1; ca < (layerCores - (p-1)); ca++) {
                        for(uint64_t f = 1; f < (std::min(layerCores - (p-1) - (ca-1), F0)); f++) {
                            for(uint64_t w = 1; w < layerCores - (p-1) - (f-1) - (ca-1); w++) {
                                ModelParams params(p, ca, f, w);
                                if(this->isValid(layerId, params)) {
                                    this->validTopologies.at(layerId).push_back(params);
                                }
                            }
                        }
                    }
                }
            }
        }

        void DataflowExplorer::printValidTopologies() {
            std::vector<std::string> names(this->layerNameToID.size(), "");

            std::map<std::string, uint64_t>::iterator it;
            for(it = this->layerNameToID.begin(); it != this->layerNameToID.end(); it++) {
                names[it->second] = it->first;
            }

            for(uint64_t i = 0; i < this->validTopologies.size(); i++) {
                llvm::outs() << "Layer: " << names[i] << " \n";
                for(ModelParams elem : this->validTopologies.at(i)) {
                    elem.print();
                }
            }
        }

        void DataflowExplorer::dumpValidTopologies() {
            std::vector<std::string> names(this->layerNameToID.size(), "");

            std::map<std::string, uint64_t>::iterator it;
            for(it = this->layerNameToID.begin(); it != this->layerNameToID.end(); it++) {
                names[it->second] = it->first;
            }

            std::ofstream configs;
            configs.open("../configs.csv");
            configs << "layerName P Ca L W Mem Compute ActCommunication WeightCommunication TotalTime\n";
            for(uint64_t i = 0; i < this->validTopologies.size(); i++) {
                llvm::outs() << "Layer: " << names[i] << " \n";
                for(ModelParams elem : this->validTopologies.at(i)) {
                    uint64_t mem = this->getTotalMemBanks(i, elem);
                    uint64_t compute = this->getComputeTime(i, elem);
                    uint64_t actComm = this->getActCommunicationTime(i, elem);
                    uint64_t wComm = this->getWeightCommunicationTime(i, elem);
                    uint64_t totalTime = this->getTotalTime(i, elem);
                    configs << names[i] << " " << elem.P << " " << elem.Ca << " " << elem.L << " " << elem.W << " "
                            << mem << " " << compute << " " << actComm << " " << wComm << " " << totalTime << "\n";
                }
            }
            configs.close();
        }
    }
}
