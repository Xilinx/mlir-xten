#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "AirDataflowExplorer.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"

#include "ATenOpReport.h"

#define DEBUG_TYPE "air-dataflow-explorer"

#define MARGIN 2

namespace xilinx {
    namespace air {
        // TODO for now both types are the same elementType
        DataflowExplorer::DataflowExplorer(std::vector<std::pair<std::string, AbsOpWrapper*>> &nameToOps) {
            unsigned int id = 0;
            for(auto pair : nameToOps) {
                this->layerNameToID[pair.first] = id;
                this->layerNameToOps.push_back(pair.second);
                id++;
            }

            this->validTopologies = std::vector<std::vector<ModelParams>>(id, std::vector<ModelParams>());
            unsigned int aWidth = this->layerNameToOps[0]->getInput().getType().dyn_cast<ShapedType>().getElementTypeBitWidth() / 8;
            this->arch = new AIEv1(aWidth, aWidth);
        }

        DataflowExplorer::~DataflowExplorer() {
            delete this->arch;
        }

        // Analytical model functions

        unsigned int DataflowExplorer::getLinesPerTile(unsigned int layerId, ModelParams &params) {
            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t linesPerBanks = (int64_t)floor(this->arch->getBankSize() / (float)lineSize);
            return linesPerBanks;
        }

        unsigned int DataflowExplorer::getBanksPerLine(unsigned int layerId, ModelParams& params) {
            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t banksPerLine = (int64_t)ceil((float)lineSize / this->arch->getBankSize());
            return banksPerLine;
        }

        // Analytical model functions
        unsigned int DataflowExplorer::getActivationInBanks(unsigned int layerId, ModelParams &params) {
            unsigned int F0 = this->layerNameToOps[layerId]->getKernelSize();
            int64_t linesPerBanks = this->getLinesPerTile(layerId, params);
            if(linesPerBanks != 0) {
                return ceil(ceil((float)F0 / params.L) / linesPerBanks);
            } else {
                unsigned int banksPerLine = this->getBanksPerLine(layerId, params);
                return ceil((float)F0 / params.L) * banksPerLine;
            }
        }

        unsigned int DataflowExplorer::getActivationOutBanks(unsigned int layerId, ModelParams &params) {
            return 2; // simple 4KB PingPong buffers
        }

        // either 2 or 4
        unsigned int DataflowExplorer::getWeightBanks(unsigned int layerId, ModelParams &params) {
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

        unsigned int DataflowExplorer::getTotalMemBanks(unsigned int layerId, ModelParams &params) {
            unsigned int inBanks = this->getActivationInBanks(layerId, params);
            unsigned int outBanks = this->getActivationOutBanks(layerId, params);
            unsigned int weightBanks = this->getWeightBanks(layerId, params);

            return inBanks + outBanks + weightBanks;
        }

        unsigned int DataflowExplorer::getMissmatchChannels(int64_t dim, unsigned int param) {
            unsigned int allGet = floor((float)dim / param) / 8;
            unsigned int someGet = dim / 8 - allGet * param;
            return someGet;
        }

        unsigned int DataflowExplorer::getMissmatchLines(int64_t dim, unsigned int param) {
            unsigned int allGet = floor((float)dim / param);
            unsigned int someGet = dim - allGet * param;
            return someGet;
        }

        unsigned int DataflowExplorer::getK(unsigned int layerId, ModelParams &params) {
            unsigned int linesPerTile = this->getLinesPerTile(layerId, params);

            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t N = aShapeAR[N_LOC];

            unsigned int K = N / linesPerTile;
            return K;
        }

        unsigned int DataflowExplorer::getComputeTimePerTile(unsigned int layerId, ModelParams &params) {
            unsigned int K = this->getK(layerId, params);
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

        unsigned int DataflowExplorer::getComputeTime(unsigned int layerId, ModelParams &params) {
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

            unsigned int missmatchCa = getMissmatchChannels(CIn, params.Ca);
            unsigned int missmatchP = getMissmatchChannels(COut, params.P);
            unsigned int missmatchL = getMissmatchLines(F, params.L);

            // TODO double check this expression
            // TODO what about efficicency here?
            unsigned int time  = macs / ((params.P - missmatchP) * (params.Ca - missmatchCa) * (params.L - missmatchL) * params.W);
            return time / this->arch->getVectSize();
        }

        unsigned int DataflowExplorer::getActCommunicationTimePerTile(unsigned int layerId, ModelParams &params) {
            unsigned int K = this->getK(layerId, params);
            return this->getActCommunicationTime(layerId, params) / K;
        }

        unsigned int DataflowExplorer::getActCommunicationTime(unsigned int layerId, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerId];
            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            uint64_t C = aShapeAR[C_LOC];
            uint64_t M = aShapeAR[M_LOC];
            uint64_t N = aShapeAR[N_LOC];

            int64_t actSize = C * M * N * aShape.getElementTypeBitWidth() / 8;
            return actSize / this->arch->getComSpeed();
        }

        unsigned int DataflowExplorer::getWeightCommunicationTimePerTile(unsigned int layerId, ModelParams &params) {
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

        unsigned int DataflowExplorer::getWeightCommunicationTime(unsigned int layerId, ModelParams &params) {
            unsigned int comPerTile = this->getWeightCommunicationTimePerTile(layerId, params);
            unsigned int K = this->getK(layerId, params);
            return comPerTile * K;
        }

        unsigned int DataflowExplorer::getTotalTimePerTile(unsigned int layerId, ModelParams &params) {
            unsigned int weightComTile = this->getWeightCommunicationTimePerTile(layerId, params);
            unsigned int actComTile = this->getActCommunicationTimePerTile(layerId, params);
            unsigned int computeTile = this->getComputeTimePerTile(layerId, params);

            // finds the bottleneck
            return std::max(std::max(actComTile, weightComTile), computeTile);
        }

        unsigned int DataflowExplorer::getTotalTime(unsigned int layerId, ModelParams &params) {
            unsigned int totalTimeTile = this->getTotalTimePerTile(layerId, params);
            unsigned int K = this->getK(layerId, params);

            return K * totalTimeTile;
        }

        // Explore functions
        std::vector<uint64_t>  DataflowExplorer::generateExplorationBounds() {
            unsigned int numCores = this->arch->getNumCores();
            std::vector<uint64_t> macsPerLayer;

            uint64_t sum = 0;
            for(AbsOpWrapper* elem : this->layerNameToOps) {
                std::map<std::string, uint64_t> stats = getStats(elem->getUnderlyingOperation());
                uint64_t macs = (stats.count("ops:MAC") == 0) ? stats["ops:>"] : stats["ops:MAC"];

                macsPerLayer.push_back(macs);
                sum += macs;
            }

            for(unsigned int i = 0; i < macsPerLayer.size(); i++) {
                macsPerLayer[i] = (macsPerLayer[i] * MARGIN / sum) * numCores;
            }

            //std::transform(macsPerLayer.begin(), macsPerLayer.end(), macsPerLayer.begin(),
            //               [&sum, &numCores](uint64_t macs) -> unsigned int {return (macs / sum) * MARGIN * numCores;});

            return macsPerLayer;
        }

        // Is not valid if does not fit under the memory constraints
        bool DataflowExplorer::isValid(unsigned int layerId, ModelParams &params) {
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

            unsigned int F0 = layer->getKernelSize();

            bool memFit = this->getTotalMemBanks(layerId, params) > this->arch->getNumBanks();
            bool enoughCIn = (CIn / params.Ca) >= 8;
            bool enoughCOut = (COut / params.P) >= 8;
            bool enoughF = (F0 / params.L) >= 1;

            return memFit && enoughCIn && enoughCOut && enoughF;
        }

        void DataflowExplorer::generateValidTopologies() {
            std::vector<uint64_t> bounds = this->generateExplorationBounds();

            for(unsigned int layerId = 0; layerId < bounds.size(); layerId++) {
                unsigned int layerCores = bounds.at(layerId);
                unsigned int F0 = this->layerNameToOps[layerId]->getKernelSize();
                for(unsigned int p = 1; p < layerCores; p++) {
                    for(unsigned int ca = 1; ca < (layerCores - (p-1)); ca++) {
                        for(unsigned int f = 1; f < (std::min(layerCores - (p-1) - (ca-1), F0)); f++) {
                            for(unsigned int w = 1; w < layerCores - (p-1) - (f-1) - (ca-1); w++) {
                                ModelParams params(p, ca, f, 1);
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
            // TODO
        }
    }
}
