#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "AirDataflowExplorer.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"

#include "ATenOpReport.h"

#define DEBUG_TYPE "air-dataflow-explorer"

namespace xilinx {
    namespace air {
        DataflowExplorer::DataflowExplorer(std::map<std::string, AbsOpWrapper*> &nameToOps) {
            this->layerNameToOps = nameToOps;
        }

        DataflowExplorer::~DataflowExplorer() {
            delete this->arch;
        }

        unsigned int DataflowExplorer::getLinesPerTile(std::string layerName, ModelParams &params) {
            ShapedType aShape = this->layerNameToOps[layerName]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t linesPerBanks = (int64_t)floor(this->arch->getBankSize() / (float)lineSize);
            return linesPerBanks;
        }

        unsigned int DataflowExplorer::getBanksPerLine(std::string layerName, ModelParams& params) {
            ShapedType aShape = this->layerNameToOps[layerName]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t C = aShapeAR[C_LOC];
            int64_t M = aShapeAR[M_LOC];

            int64_t lineSize = (ceil(C / params.P) * M) * (aShape.getElementTypeBitWidth() / 8);
            int64_t banksPerLine = (int64_t)ceil((float)lineSize / this->arch->getBankSize());
            return banksPerLine;
        }

        // Analytical model functions
        unsigned int DataflowExplorer::getActivationInBanks(std::string layerName, ModelParams &params) {
            unsigned int F0 = this->layerNameToOps[layerName]->getKernelSize();
            int64_t linesPerBanks = this->getLinesPerTile(layerName, params);
            if(linesPerBanks != 0) {
                return ceil(ceil((float)F0 / params.L) / linesPerBanks);
            } else {
                unsigned int banksPerLine = this->getBanksPerLine(layerName, params);
                return ceil((float)F0 / params.L) * banksPerLine;
            }
        }

        unsigned int DataflowExplorer::getActivationOutBanks(std::string layer, ModelParams &params) {
            return 2; // simple 4KB PingPong buffers
        }

        // either 2 or 4
        unsigned int DataflowExplorer::getWeightBanks(std::string layer, ModelParams &params) {
            if(!this->layerNameToOps[layer]->hasWeights()) {
                return 0;
            }

            ShapedType wShape = this->layerNameToOps[layer]->getWeights().getType().dyn_cast<ShapedType>();
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

        unsigned int DataflowExplorer::getTotalMemBanks(std::string layer, ModelParams &params) {
            unsigned int inBanks = this->getActivationInBanks(layer, params);
            unsigned int outBanks = this->getActivationOutBanks(layer, params);
            unsigned int weightBanks = this->getWeightBanks(layer, params);

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

        unsigned int DataflowExplorer::getK(std::string layerName, ModelParams &params) {
            unsigned int linesPerTile = this->getLinesPerTile(layerName, params);

            ShapedType aShape = this->layerNameToOps[layerName]->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t N = aShapeAR[N_LOC];

            unsigned int K = N / linesPerTile;
            return K;
        }

        unsigned int DataflowExplorer::getComputeTimePerTile(std::string layerName, ModelParams &params) {
            unsigned int K = this->getK(layerName, params);
            return this->getComputeTime(layerName, params) / K;
        }

        unsigned int DataflowExplorer::getComputeTime(std::string layerName, ModelParams &params) {
            std::map<std::string, uint64_t> stats = xilinx::aten::getATenOpStats(this->layerNameToOps[layerName]->getUnderlyingOperation());
            uint64_t macs = stats["ops:MAC"];

            AbsOpWrapper* layer = this->layerNameToOps[layerName];
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

        unsigned int DataflowExplorer::getActCommunicationTimePerTile(std::string layer, ModelParams &params) {
            unsigned int K = this->getK(layer, params);
            return this->getActCommunicationTime(layer, params) / K;
        }

        unsigned int DataflowExplorer::getActCommunicationTime(std::string layerName, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerName];
            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            uint64_t C = aShapeAR[C_LOC];
            uint64_t M = aShapeAR[M_LOC];
            uint64_t N = aShapeAR[N_LOC];

            int64_t actSize = C * M * N * aShape.getElementTypeBitWidth() / 8;
            return actSize / this->arch->getComSpeed();
        }

        unsigned int DataflowExplorer::getWeightCommunicationTimePerTile(std::string layer, ModelParams &params) {
            if(!this->layerNameToOps[layer]->hasWeights()) {
                return 0;
            }

            ShapedType wShape = this->layerNameToOps[layer]->getWeights().getType().dyn_cast<ShapedType>();
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

        unsigned int DataflowExplorer::getWeightCommunicationTime(std::string layerName, ModelParams &params) {
            unsigned int comPerTile = this->getWeightCommunicationTimePerTile(layerName, params);
            unsigned int K = this->getK(layerName, params);
            return comPerTile * K;
        }

        unsigned int DataflowExplorer::getTotalTimePerTile(std::string layerName, ModelParams &params) {
            unsigned int weightComTile = this->getWeightCommunicationTimePerTile(layerName, params);
            unsigned int actComTile = this->getActCommunicationTimePerTile(layerName, params);
            unsigned int computeTile = this->getComputeTimePerTile(layerName, params);

            // finds the bottleneck
            return std::max(std::max(actComTile, weightComTile), computeTile);
        }

        unsigned int DataflowExplorer::getTotalTime(std::string layerName, ModelParams &params) {
            unsigned int totalTimeTile = this->getTotalTimePerTile(layerName, params);
            unsigned int K = this->getK(layerName, params);

            return K * totalTimeTile;
        }
    }
}
