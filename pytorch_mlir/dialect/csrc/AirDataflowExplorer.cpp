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
#define DW_SHARED true
#define MIN_BOUND 8

// TODO also take into account kernel efficiency?
// TODO Need to take into account kernel fusion at some point, either here or afterwards
// TODO Need to incorporate external bandwdith at some point
// TODO take into account parallel line handling when memory or no?
// TODO Check that utilization is computed correctly, or latency etc..
// TODO also implement generic one without any architecture restructions?
// TODO Double check W == W

// TODO Check that tile bring time is correct when need to bring multiple tiles at once
// TODO always ensure that the forwarded tile can be sent without any trouble

namespace xilinx {
    namespace air {
        // TODO for now both types are the same elementType
        DataflowExplorer::DataflowExplorer(std::vector<std::pair<std::string, AbsOpWrapper*>> &nameToOps) {
            uint64_t id = 0;
            for(auto pair : nameToOps) {
                this->layerNameToID[pair.first] = id;
                this->layerIdToName[id] = pair.first;
                this->layerNameToOps.push_back(pair.second);
                id++;
            }

            this->validTopologies = std::vector<std::vector<ModelParams>>(id, std::vector<ModelParams>());
            ShapedType aShape = this->layerNameToOps[0]->getInput().getType().dyn_cast<ShapedType>();
            uint64_t aWidth = getElementWidth(aShape, FORCE_INT8);
            this->arch = new AIEv1(aWidth, aWidth);
        }

        DataflowExplorer::~DataflowExplorer() {
            delete this->arch;
        }

        // Analytical model functions

        // If aShapeIn has been provided, then work from there but assume split already occured
        // TODO this is not ideal, clean that up with proper pre / post transformation
        uint64_t DataflowExplorer::getLinesPerTile(uint64_t layerId, ModelParams &params, llvm::Optional<ShapedType> aShapeIn) {
            if(params.P == 0 || params.Ca == 0 || params.L == 0 || params.W == 0) {
                llvm::outs() << "params was 0 in getLinesPerTile...\n";
            }

            if(aShapeIn.hasValue()) {
                ShapedType aShape = aShapeIn.getValue();

                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                int64_t C = aShapeAR[C_LOC];
                int64_t M = aShapeAR[M_LOC];
                //int64_t N = aShapeAR[N_LOC];

                int64_t lineSize = (C * M) * getElementWidth(aShape, FORCE_INT8);
                int64_t linesPerBanks = (int64_t)floor(this->arch->getBankSize() / (float)lineSize);

                if(params.lineGranularity) {
                    return linesPerBanks >= 1 ? 1 : 0;
                } else {
                    return linesPerBanks;
                }
            } else {
                ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                int64_t C = aShapeAR[C_LOC];
                int64_t M = aShapeAR[M_LOC];
                //int64_t N = aShapeAR[N_LOC];

                int64_t lineSize = (ceil(C / params.Ca) * M) * getElementWidth(aShape, FORCE_INT8);
                int64_t linesPerBanks = (int64_t)floor(this->arch->getBankSize() / (float)lineSize);

                if(params.lineGranularity) {
                    return linesPerBanks >= 1 ? 1 : 0;
                } else {
                    return linesPerBanks;
                }
            }

        }

        // TODO Double check that function
        // TODO maybe add one for the tail of a Cascade chain
        // TODO rename to getMaxTilesPerCore
        uint64_t DataflowExplorer::getTilesPerCore(uint64_t layerId, ModelParams &params, llvm::Optional<ShapedType> aShapeIn,
                                                   llvm::Optional<uint64_t> F0In) {
            uint64_t linesPerTile = this->getLinesPerTile(layerId, params, aShapeIn);
            uint64_t banksPerLine = this->getBanksPerLine(layerId, params, aShapeIn);

            uint64_t F0OverF;
            uint64_t F0;
            if(F0In.hasValue()) {
                // Used in WSplit only so has already only the part of weight of interest
                assert(aShapeIn.hasValue());
                F0 = F0In.getValue();
                F0OverF = F0;
            } else {
                F0 = this->layerNameToOps[layerId]->getKernelSize();
                F0OverF = ceil((float)F0 / params.L);
            }

            uint64_t producedLines = linesPerTile;
            uint64_t worstLocLines = (F0OverF - 1) + producedLines;

            if(linesPerTile != 0) {
                if(params.lineGranularity) {
                    return F0OverF;
                } else {
                    return 1 + ceil((float)(worstLocLines - 1) / linesPerTile);
                }
            } else {
                return F0OverF * banksPerLine;
            }
        }

        uint64_t DataflowExplorer::getBanksPerLine(uint64_t layerId, ModelParams& params, llvm::Optional<ShapedType> aShapeIn) {
            if(aShapeIn.hasValue()) {
                ShapedType aShape = aShapeIn.getValue();

                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                int64_t C = aShapeAR[C_LOC];
                int64_t M = aShapeAR[M_LOC];

                int64_t lineSize = (C * M) * getElementWidth(aShape, FORCE_INT8);
                int64_t banksPerLine = (int64_t)ceil((float)lineSize / this->arch->getBankSize());

                return banksPerLine;
            } else {
                ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                int64_t C = aShapeAR[C_LOC];
                int64_t M = aShapeAR[M_LOC];

                int64_t lineSize = (ceil(C / params.Ca) * M) * getElementWidth(aShape, FORCE_INT8);
                int64_t banksPerLine = (int64_t)ceil((float)lineSize / this->arch->getBankSize());

                return banksPerLine;
            }
        }

        uint64_t DataflowExplorer::getK(uint64_t layerId, ModelParams &params) {
            uint64_t linesPerTile = this->getLinesPerTile(layerId, params, llvm::Optional<ShapedType>());

            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t N = aShapeAR[N_LOC];

            uint64_t K = std::max((uint64_t)1, (uint64_t)ceil((float)N / linesPerTile));

            //llvm::outs() << "N= " << N << " K = " << K << " linesPerTile = " << linesPerTile <<"\n";
            return K;
        }

        // Analytical model functions
        uint64_t DataflowExplorer::getActivationInBanks(uint64_t layerId, ModelParams &params) {
            uint64_t F0 = this->layerNameToOps[layerId]->getKernelSize();
            int64_t linesPerBanks = this->getLinesPerTile(layerId, params, llvm::Optional<ShapedType>());
            uint64_t banksPerLine = this->getBanksPerLine(layerId, params, llvm::Optional<ShapedType>());

            ShapedType aShape = this->layerNameToOps[layerId]->getInput().getType().dyn_cast<ShapedType>();

            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            int64_t N = aShapeAR[N_LOC];
            bool allLinesIn = (N <= linesPerBanks);

            uint64_t banksForFilter = this->getTilesPerCore(layerId, params, llvm::Optional<ShapedType>(),
                                                            llvm::Optional<uint64_t>());

            uint64_t minBanksForFilter = (F0 == 1) ? 1 : (allLinesIn ? 1 : 2);
            return std::max(minBanksForFilter, banksForFilter) + banksPerLine;
        }

        // PingPong 4KB buffers
        // If Cascade is there can use the shared memory between the cores to share output space
        // TODO this is architecture specific, make that more generic
        uint64_t DataflowExplorer::getActivationOutBanks(uint64_t layerId, ModelParams &params) {
            if(params.Ca == 1 && params.L == 1) {
                return 2;
            } else {
                return 1;
            }
        }

        // either 2 or 4
        // TODO check that not in F duplication case
        // TODO how do we handle biases?
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

            //int64_t weightSize = COut * CIn * F0 * F1 * getElementWidth(wShape, FORCE_INT8);

            int64_t locCout = ceil((float)COut / params.P);
            int64_t locCin = ceil((float)CIn / params.Ca);
            int64_t locF0 = ceil((float)F0 / params.L);
            int64_t locWeightSize = locCout * locCin * locF0 * F1 * getElementWidth(wShape, FORCE_INT8);

            int64_t weightBanks = ceil(locWeightSize / this->arch->getBankSize());

            if((weightBanks > 4) || (weightBanks == 3)) {
                return 4;
            } else {
                return 2;
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

        uint64_t DataflowExplorer::getComputeTimePerTile(uint64_t layerId, ModelParams &params) {
            if(params.lineGranularity) {
                AbsOpWrapper* layer = this->layerNameToOps[layerId];
                ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                uint64_t N = aShapeAR[N_LOC];

                return this->getComputeTime(layerId, params) / N;
            } else {
                uint64_t K = this->getK(layerId, params);
                return this->getComputeTime(layerId, params) / K;
            }
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
            //llvm::outs() << "macs should be: " << macs << "and time is: " << time <<"\n";
            //uint64_t kernelEfficiency = this->layerNameToOps[layerId]->getKernelEfficiency(this->arch);
            uint64_t kernelEfficiency = 1; // TODO fix function call..
            return time / (this->arch->getVectSize() * kernelEfficiency);
        }

        uint64_t DataflowExplorer::getActCommunicationTimePerTile(uint64_t layerId, ModelParams &params) {
            uint64_t comTime = this->getActCommunicationTime(layerId, params);

            if(params.lineGranularity) {
                AbsOpWrapper* layer = this->layerNameToOps[layerId];
                ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                uint64_t N = aShapeAR[N_LOC];

                return comTime / N;
            } else {
                uint64_t K = this->getK(layerId, params);
                return comTime / K;
            }

        }

        uint64_t DataflowExplorer::getActCommunicationTime(uint64_t layerId, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerId];
            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();
            uint64_t C = aShapeAR[C_LOC];
            uint64_t M = aShapeAR[M_LOC];
            uint64_t N = aShapeAR[N_LOC];

            int64_t actSize = C * M * N * getElementWidth(aShape, FORCE_INT8);

            if(DW_SHARED &&
               ((this->layerNameToOps[layerId]->isDepthWise() && (layerId > 0) && !this->layerNameToOps[layerId-1]->isDepthWise())
                || ((layerId == 0) && this->layerNameToOps[layerId]->isDepthWise()))) {
                actSize = 0; // TODO make better, to make com 0 because assume shared memory
            }

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

            //int64_t weightSize = COut * CIn * F0 * F1 * getElementWidth(wShape, FORCE_INT8);

            int64_t locCout = ceil((float)COut / params.P);
            int64_t locCin = ceil((float)CIn / params.Ca);
            int64_t locF0 = ceil((float)F0 / params.L);
            int64_t locWeightSize = locCout * locCin * locF0 * F1 * getElementWidth(wShape, FORCE_INT8);

            int64_t weightBanks = ceil(locWeightSize / this->arch->getBankSize());

            if(weightBanks <= 4) {
                return 0;
            }

            return locWeightSize / this->arch->getComSpeed();
        }

        uint64_t DataflowExplorer::getWeightCommunicationTime(uint64_t layerId, ModelParams &params) {
            uint64_t comPerTile = this->getWeightCommunicationTimePerTile(layerId, params);

            if(params.lineGranularity) {
                AbsOpWrapper* layer = this->layerNameToOps[layerId];
                ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                uint64_t N = aShapeAR[N_LOC];

                return comPerTile * N;
            } else {
                uint64_t K = this->getK(layerId, params);
                return comPerTile * K;
            }
        }

        uint64_t DataflowExplorer::getTotalTimePerTile(uint64_t layerId, ModelParams &params) {
            uint64_t weightComTile = this->getWeightCommunicationTimePerTile(layerId, params);
            uint64_t actComTile = this->getActCommunicationTimePerTile(layerId, params);
            uint64_t computeTile = this->getComputeTimePerTile(layerId, params);

            // finds the bottleneck
            return std::max(std::max(actComTile, weightComTile), computeTile);
        }

        uint64_t DataflowExplorer::getTotalTime(uint64_t layerId, ModelParams &params) {
            if(params.lineGranularity) {
                AbsOpWrapper* layer = this->layerNameToOps[layerId];
                ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> aShapeAR = aShape.getShape();
                uint64_t N = aShapeAR[N_LOC];

                uint64_t totalTimeTile = this->getTotalTimePerTile(layerId, params);

                return N * totalTimeTile;
            } else {
                uint64_t totalTimeTile = this->getTotalTimePerTile(layerId, params);
                uint64_t K = this->getK(layerId, params);

                return K * totalTimeTile;
            }
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
            if(params.size() == 0) {
                return (uint64_t)-1;
            }

            uint64_t locSlowest = 0;
            uint64_t slowest = 0;
            uint64_t loc = 0;
            for(uint64_t i = 0; i < params.size(); i++) {
                if((params.at(i).P != 0) && (params.at(i).Ca != 0) && (params.at(i).L != 0) && (params.at(i).W != 0)) {
                    uint64_t totalTimeLayer = this->getTotalTime(loc, params.at(i));
                    if(totalTimeLayer > slowest) {
                        slowest = totalTimeLayer;
                        locSlowest = i;
                    }
                    loc++;
                }
            }

            uint64_t latency = 0;
            loc = 0;
            for(uint64_t i = 0; i < locSlowest; i++) {
                if((params.at(i).P != 0) && (params.at(i).Ca != 0) && (params.at(i).L != 0) && (params.at(i).W != 0)) {
                    latency += this->getTotalTimePerTile(loc, params.at(i));
                    loc++;
                }
            }

            latency += this->getTotalTime(loc, params.at(locSlowest));
            loc++;

            for(uint64_t i = locSlowest + 1; i < params.size(); i++) {
                if((params.at(i).P != 0) && (params.at(i).Ca != 0) && (params.at(i).L != 0) && (params.at(i).W != 0)) {
                    latency += this->getTotalTimePerTile(loc, params.at(i));
                    loc++;
                }
            }

            return latency;
        }

        uint64_t getThroughputFromDelay(uint64_t delay) {
            return (uint64_t)(1/(delay * pow(10, -9)));
        }

        uint64_t DataflowExplorer::getThroughput(std::vector<ModelParams> &params) {
            if(params.size() == 0) {
                return 0;
            }

            uint64_t throughput = (uint64_t)-1;
            uint64_t loc = 0;
            for(uint64_t i = 0; i < params.size(); i++) {
                if(params.at(i).P != 0 && params.at(i).Ca != 0 && params.at(i).L != 0 && params.at(i).W != 0) {
                    uint64_t totalTimeLayer = this->getTotalTime(loc, params.at(i));
                    uint64_t layerThroughput = getThroughputFromDelay(totalTimeLayer);
                    if(layerThroughput < throughput) {
                        throughput = layerThroughput;
                    }

                    loc++;
                }
            }

            return throughput;
        }

        // Computes utilizaton of whole array
        double DataflowExplorer::getUtilization(std::vector<ModelParams> &params) {
            uint64_t maxWorkDone = this->arch->getNumCores() * this->arch->getClockFrequency() * this->arch->getVectSize();
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
            std::vector<uint64_t> macsPerLayer;

            uint64_t sum = 0;
            for(AbsOpWrapper* elem : this->layerNameToOps) {
                std::map<std::string, uint64_t> stats = getStats(elem->getUnderlyingOperation());
                uint64_t macs = (stats.count("ops:MAC") == 0) ? stats["ops:>"] : stats["ops:MAC"];
                llvm::outs() << "macs were: " << macs << "\n";

                macsPerLayer.push_back(macs);
                sum += macs;
            }

            uint64_t numCores = this->arch->getNumCores();
            for(uint64_t i = 0; i < macsPerLayer.size(); i++) {
                macsPerLayer[i] = std::min((uint64_t)(((double)macsPerLayer[i] * MARGIN / sum) * numCores), numCores);
                macsPerLayer[i] = std::max(macsPerLayer[i], (uint64_t)MIN_BOUND);
            }

            //std::transform(macsPerLayer.begin(), macsPerLayer.end(), macsPerLayer.begin(),
            //               [&sum, &numCores](uint64_t macs) -> uint64_t {return (macs / sum) * MARGIN * numCores;});

            return macsPerLayer;
        }

        // Is not valid if does not fit under the memory constraints
        // TODO check with line stuff
        bool DataflowExplorer::isValid(uint64_t layerId, ModelParams &params) {
            AbsOpWrapper* layer = this->layerNameToOps[layerId];

            ShapedType aShape = layer->getInput().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t CIn = aShapeAR[C_LOC];
            int64_t N = aShapeAR[N_LOC];
            int64_t COut;
            if(layer->hasWeights()) {
                ShapedType wShape = layer->getWeights().getType().dyn_cast<ShapedType>();
                ArrayRef<int64_t> wShapeAR = wShape.getShape();
                COut = wShapeAR[COUT_LOC];
            } else {
                COut = CIn;
            }

            uint64_t F0 = layer->getKernelSize();

            bool enoughCIn = (CIn / params.Ca) >= 8;
            bool enoughCOut = (COut / params.P) >= 8;
            bool enoughF = (F0 / params.L) >= 1;
            bool enoughW = (N / params.W) >= 1;

            if(enoughCIn && enoughCOut && enoughF && enoughW) {
                bool memFit = this->getTotalMemBanks(layerId, params) <= this->arch->getNumBanks();
                return memFit;
            } else {
                return false;
            }

            // TODO also check that W is rate balanced ?

            /*if(layerId == 4) {
                llvm::outs() << "MemFit: " << memFit << " Cin: " << enoughCIn << " COut " << enoughCOut << " F " << enoughF << "\n";
                llvm::outs() << "For params: (P=" << params.P << ", Ca=" << params.Ca << ", F=" << params.L << ", W=" << params.W << "\n";
                llvm::outs() << "MemFit is: " << this->getTotalMemBanks(layerId, params) << "with: " <<
                    this->getActivationInBanks(layerId, params) << ", " <<
                    this->getActivationOutBanks(layerId, params) << ", " <<
                    this->getWeightBanks(layerId, params) << "\n";
                    }*/

            //return memFit && enoughCIn && enoughCOut && enoughF;
        }

        // In favor of tile handling when compute time is the same and both fits (same result if F == 1)
        void DataflowExplorer::generateValidTopologies() {
            std::vector<uint64_t> bounds = this->generateExplorationBounds();

            for(auto i : bounds) {
                llvm::outs() << "Bounds: " << i << "\n";
            }

            for(uint64_t layerId = 0; layerId < bounds.size(); layerId++) {
                uint64_t layerCores = bounds.at(layerId);
                uint64_t F0 = this->layerNameToOps[layerId]->getKernelSize();
                for(uint64_t p = 1; p <= layerCores; p++) {
                    for(uint64_t ca = 1; ca <= (layerCores - (p-1)); ca++) {
                        for(uint64_t f = 1; f <= (std::min(layerCores - (p-1) - (ca-1), F0)); f++) {
                            for(uint64_t w = 1; w <= layerCores - (p-1) - (f-1) - (ca-1); w++) {
                                ModelParams paramsLine(p, ca, f, w, true);
                                ModelParams paramsTile(p, ca, f, w, false);

                                bool lineValid = this->isValid(layerId, paramsLine);
                                bool tileValid = this->isValid(layerId, paramsTile);
                                if(lineValid && tileValid) {
                                    if(this->getTotalTime(layerId, paramsLine) >= this->getTotalTime(layerId, paramsTile)) {
                                        this->validTopologies.at(layerId).push_back(paramsTile);
                                    } else {
                                        this->validTopologies.at(layerId).push_back(paramsLine);
                                    }
                                } else if(tileValid) {
                                    this->validTopologies.at(layerId).push_back(paramsTile);
                                } else if(lineValid) {
                                    this->validTopologies.at(layerId).push_back(paramsLine);
                                }
                            }
                        }
                    }
                }
            }
        }

        // take valid topologies and build a graph with ins set to all nodes, areaToNode left empty
        // Also take into account communication characteristics of the underlying architecture
        void DataflowExplorer::generatePathGraph() {
            this->pathGraph = std::vector<std::vector<Node_t*>>(this->validTopologies.size() + 2, std::vector<Node_t*>());

            Node_t* root = new Node_t(ModelParams(0,0,0,0,false));
            root->areaToThroughput = std::vector<PathInfo_t>(this->arch->getNumCores() + 2, PathInfo_t((uint64_t)0));
            root->areaToLatency = std::vector<PathInfo_t>(this->arch->getNumCores() + 2, PathInfo_t((uint64_t)-1));

            // Init paths of size
            root->areaToThroughput[0].path = std::vector<ModelParams>(1, ModelParams(0,0,0,0,false));
            root->areaToLatency[0].path = std::vector<ModelParams>(1, ModelParams(0,0,0,0,false));

            this->pathGraph.at(0).push_back(root);

            for(unsigned int i = 0; i < this->validTopologies.size(); i++) {
                for(ModelParams p : this->validTopologies.at(i)) {
                    if(i == 0) {
                        Node_t* node = new Node_t(p);
                        node->ins.push_back(root);
                        this->pathGraph.at(i+1).push_back(node);
                    } else {
                        Node_t* node = new Node_t(p);

                        // TODO check ins append constraints in more details
                        for(Node_t* n : this->pathGraph.at(i)) {
                            if(this->layerNameToOps.at(i)->isDepthWise()) {
                                unsigned int nP = n->params.P;
                                unsigned int nW = n->params.W;
                                unsigned int P = p.P;
                                unsigned int L = p.L;
                                unsigned int W = p.W;

                                // Take into account communication constraints
                                if(nP == P && ((nW == 1) || (nW == L) || (nW == W))) {
                                    node->ins.push_back(n);
                                }
                            } else {
                                unsigned int nP = n->params.P;
                                unsigned int nW = n->params.W;
                                unsigned int Ca = p.Ca;
                                unsigned int L = p.L;
                                unsigned int W = p.W;

                                // Take into account communication constraints
                                if(nP == Ca && ((nW == 1) || (nW == L) || (nW == W))) {
                                    node->ins.push_back(n);
                                }
                            }
                        }

                        this->pathGraph.at(i+1).push_back(node);
                    }
                }
            }

            Node_t* sink = new Node_t(ModelParams(0,0,0,0,false));
            for(Node_t* n : this->pathGraph.at(this->pathGraph.size() - 2)) {
                //llvm::outs() << "Extending the sink...\n";
                sink->ins.push_back(n);
            }

            this->pathGraph.at(this->pathGraph.size() - 1).push_back(sink);
        }

        // Uses the ins generated by previous function to build the areaToNode for all functions
        // TODO make that function look better
        // TODO could potentially parallelize that stuf..
        // TODO maybe we should also remove some of the copies and replace with more pointers?
        void DataflowExplorer::enumeratePaths() {
            llvm::outs() << "Path Graph.size() = " << this->pathGraph.size() << "\n";

            for(uint64_t layer = 1; layer < this->pathGraph.size(); layer++) {
                llvm::outs() << "Handling layer: " << layer << "\n";
                for(Node_t* layerNode : this->pathGraph.at(layer)) {
                    layerNode->areaToThroughput = std::vector<PathInfo_t>(this->arch->getNumCores() + 2, PathInfo_t((uint64_t)0));
                    layerNode->areaToLatency = std::vector<PathInfo_t>(this->arch->getNumCores() + 2, PathInfo_t((uint64_t)-1));

                    for(Node_t* inNode : layerNode->ins) {
                        // Handle throughput
                        for(uint64_t i = 0; i < inNode->areaToThroughput.size(); i++) {
                            std::vector<ModelParams> pathHead = inNode->areaToThroughput.at(i).path;
                            if(pathHead.size() != 0) {
                                pathHead.push_back(layerNode->params);
                                uint64_t nArea = i + layerNode->params.cores();

                                if(nArea <= this->arch->getNumCores()) {
                                    uint64_t nodeTotalTime = (layer == this->pathGraph.size()-1) ? 0 : this->getTotalTime(layer-1, layerNode->params);
                                    uint64_t  nThroughput;
                                    if(nodeTotalTime > inNode->areaToThroughput.at(i).maxTotalTime) {
                                        nThroughput = this->getThroughput(pathHead);
                                    } else {
                                        nThroughput = inNode->areaToThroughput.at(i).value;
                                        nodeTotalTime = inNode->areaToThroughput.at(i).maxTotalTime;
                                    }

                                    uint64_t locThroughput = layerNode->areaToThroughput.at(nArea).value;
                                    if(nThroughput > locThroughput) {
                                        layerNode->areaToThroughput[nArea] = PathInfo_t((uint64_t)0); // TODO double check that assignment
                                        layerNode->areaToThroughput[nArea].path = pathHead;
                                        layerNode->areaToThroughput[nArea].value = nThroughput;
                                        layerNode->areaToThroughput[nArea].maxTotalTime = nodeTotalTime;
                                    }
                                }
                            }
                        }

                        // Handle Latency
                       for(uint64_t i = 0; i < inNode->areaToLatency.size(); i++) {
                            std::vector<ModelParams> pathHead = inNode->areaToLatency.at(i).path;
                            if(pathHead.size() != 0) {
                                pathHead.push_back(layerNode->params);
                                uint64_t nArea = i + layerNode->params.cores();

                                if(nArea <= this->arch->getNumCores()) {
                                    uint64_t nodeTotalTime = (layer == this->pathGraph.size()-1) ? 0 : this->getTotalTime(layer-1, layerNode->params);

                                    uint64_t nLatency;
                                    if(nodeTotalTime > inNode->areaToLatency.at(i).maxTotalTime) {
                                        nLatency = this->getEndToEndLatency(pathHead);
                                    } else {
                                        uint64_t totalTimeTile = (layer == this->pathGraph.size()-1) ? 0 : this->getTotalTimePerTile(layer-1, layerNode->params);
                                        nLatency = inNode->areaToLatency.at(i).value + totalTimeTile;
                                        nodeTotalTime = inNode->areaToLatency.at(i).maxTotalTime;
                                    }

                                    uint64_t locLatency = layerNode->areaToLatency.at(nArea).value;
                                    if(nLatency < locLatency) {
                                        layerNode->areaToLatency[nArea] = PathInfo_t((uint64_t)-1);
                                        layerNode->areaToLatency[nArea].path = pathHead; // TODO double check that assignment
                                        layerNode->areaToLatency[nArea].value = nLatency;
                                        layerNode->areaToLatency[nArea].maxTotalTime = nodeTotalTime;
                                    }
                                }
                            }
                       }
                    }
                }
            }
        }

        void DataflowExplorer::getParetoFrontierAndCleanGraph() {
            for(uint64_t l = 0; l < this->pathGraph.size(); l++) {
                std::vector<Node_t*> layer = this->pathGraph.at(l);

                if(l == (this->pathGraph.size() - 1)) {
                    assert(layer.size() == 1);
                    //llvm::outs() << "When collecting got: " << layer.at(0)->areaToThroughput.size() << "\n";
                    //for(uint64_t i = 0; i < layer.at(0)->areaToThroughput.size(); i++) {
                        //llvm::outs() << i << ": " << layer.at(0)->areaToThroughput.at(i).size() << "\n";
                    //}

                    this->paretoThroughput = layer.at(0)->areaToThroughput;
                    this->paretoLatency = layer.at(0)->areaToLatency;
                }

                for(uint64_t n = 0; n < layer.size(); n++) {
                    Node_t* node = layer.at(n);

                    node->ins.clear();
                    delete node;
                }
            }
        }

        void DataflowExplorer::enumerate() {
            this->generateValidTopologies();
            this->dumpValidTopologies();
            this->generatePathGraph();
            this->enumeratePaths();
            this->getParetoFrontierAndCleanGraph();
        }

        // Visualisations stuff

        void DataflowExplorer::printValidTopologies() {
            llvm::outs() << "Valid topologies size: " << this->validTopologies.size() << "\n";
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

        void DataflowExplorer::dumpModelParam(ModelParams &params, std::ofstream &outputFile, std::string layerName, uint64_t i) {
            uint64_t K = this->getK(i, params);
            uint64_t mem = this->getTotalMemBanks(i, params);
            uint64_t compute = this->getComputeTime(i, params);
            uint64_t actComm = this->getActCommunicationTime(i, params);
            uint64_t wComm = this->getWeightCommunicationTime(i, params);
            uint64_t totalTime = this->getTotalTime(i, params);
            uint64_t memActIn = this->getActivationInBanks(i, params);
            uint64_t memActOut = this->getActivationOutBanks(i, params);
            uint64_t memWeights = this->getWeightBanks(i, params);
            outputFile << layerName << " " << params.P << " " << params.Ca << " " << params.L << " " << params.W << " " <<
                K << " "<< mem << " " << compute << " " << actComm << " " << wComm << " " << totalTime << " " <<
                memActIn << " " << memActOut << " " << memWeights << "\n";

        }

        void DataflowExplorer::dumpValidTopologies() {
            std::vector<std::string> names(this->layerNameToID.size(), "");
            llvm::outs() << "ValidTopologies size: " << this->validTopologies.size() << " namesSize " << names.size() << "\n";

            std::map<std::string, uint64_t>::iterator it;
            for(it = this->layerNameToID.begin(); it != this->layerNameToID.end(); it++) {
                names[it->second] = it->first;
            }

            std::ofstream configs;
            configs.open("./output/configs.csv", std::ios::out);
            configs << "layerName P Ca L W K Mem Compute ActCommunication WeightCommunication TotalTime MemActIn MemActOut MemWeight\n";

            for(uint64_t i = 0; i < this->validTopologies.size(); i++) {
                llvm::outs() << "Layer: " << names[i] << ", with valid topologies: " << this->validTopologies.at(i).size() << "\n";
                for(ModelParams elem : this->validTopologies.at(i)) {
                    this->dumpModelParam(elem, configs, names[i], i);
                }
            }
            configs.close();
        }

        void DataflowExplorer::dumpParetoFrontiers() {
            std::ofstream pareto;
            pareto.open("./output/pareto_throughput.csv", std::ios::out);
            pareto << "Area Throughput Utilization\n";

            for(uint64_t i = 0; i < this->paretoThroughput.size(); i++) {
                if(this->paretoThroughput.at(i).path.size() != 0) {
                    //llvm::outs() << "For area: " << i << " has " << this->paretoThroughput.at(i).value << "\n";
                    //for(uint64_t j = 0; j < this->paretoThroughput.at(i).size(); j++) {
                    //    this->paretoThroughput.at(i).at(j).print();
                    //}
                    pareto << i << " " << this->paretoThroughput.at(i).value << " "
                           << this->getUtilization(this->paretoThroughput.at(i).path) << "\n";
                }
            }

            pareto.close();

            pareto.open("./output/pareto_latency.csv", std::ios::out);
            pareto << "Area Latency\n";

            for(uint64_t i = 0; i < paretoLatency.size(); i++) {
                if(paretoLatency.at(i).path.size() != 0) {
                    //llvm::outs() << "For area: " << i << " has " << this->getEndToEndLatency(this->paretoLatency.at(i)) << "\n";
                    //for(uint64_t j = 0; j < this->paretoLatency.at(i).size(); j++) {
                    //    this->paretoLatency.at(i).at(j).print();
                    //}
                    pareto << i << " " << paretoLatency.at(i).value << " "
                           << this->getUtilization(this->paretoLatency.at(i).path) << "\n";
                }
            }

            pareto.close();
        }

        void DataflowExplorer::dumpPath(PathInfo_t &path, std::string fname) {
            std::vector<std::string> names(this->layerNameToID.size(), "");
            std::map<std::string, uint64_t>::iterator it;
            for(it = this->layerNameToID.begin(); it != this->layerNameToID.end(); it++) {
                names[it->second] = it->first;
            }

            std::ofstream outF;
            outF.open(fname, std::ios::out);
            outF << "layerName P Ca L W K Mem Compute ActCommunication WeightCommunication TotalTime MemActIn MemActOut MemWeight\n";

            uint64_t loc = 0;
            for(ModelParams p : path.path) {
                if(p.nonZero()) {
                    this->dumpModelParam(p, outF, names[loc], loc);
                    loc++;
                }

            }

            outF.close();
        }

        void DataflowExplorer::dumpPathsFrom(std::vector<PathInfo_t> &paths, std::string prefix) {
            for(uint64_t i = 0; i < paths.size(); i++) {
                if(paths.at(i).path.size() != 0) {
                    this->dumpPath(paths.at(i), prefix + std::to_string(i) + ".csv");
                }
            }
        }

        std::map<std::string, ModelParams> DataflowExplorer::getMaxThroughput() {
            if(this->paretoThroughput.size() == 0) {
                llvm::outs() << "Must run the exploration first before extracting the maximum throughput..\n";
                return std::map<std::string, ModelParams>();
            }

            uint64_t maxValue = 0;
            std::vector<ModelParams> bestPath;
            for(auto pathInfo : this->paretoThroughput) {
                if(pathInfo.value > maxValue) {
                    maxValue = pathInfo.value;
                    bestPath = pathInfo.path;
                }
            }

            llvm::outs() << "Using: \n";
            for(ModelParams p : bestPath) {
                p.print();
            }

            std::map<std::string, ModelParams> layerNameToParams;
            uint64_t loc = 0;
            for(uint64_t i = 0; i < bestPath.size(); i++) {
                if(bestPath.at(i).nonZero()) {
                    layerNameToParams[this->layerIdToName[loc]] = bestPath.at(i);
                    loc++;
                }
            }

            return layerNameToParams;
        }
    }
}
