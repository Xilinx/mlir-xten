//===- ATenVisualGraph.cpp -----------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Transform/ATenVisualGraph.h"
#include "PassDetail.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "aten-visual-graph"

using namespace mlir;
using namespace xilinx::xten;
using namespace xilinx;
using namespace mlir::torch;

namespace {

template <class Op>
llvm::Optional<Value> getAlpha(Op &reluOp) {
  return reluOp.getAlpha();
}

template <>
llvm::Optional<Value> getAlpha(Torch::AtenReluOp &reluOp) {
  return {};
}

template <>
llvm::Optional<Value> getAlpha(xten::Conv2dReLUOp &reluOp) {
  return {};
}

template <>
llvm::Optional<Value> getAlpha(xten::Conv2dReLUMaxPoolOp &reluOp) {
  return {};
}

template <>
llvm::Optional<Value> getAlpha(xten::Conv2dReLUPadMaxPoolOp &reluOp) {
  return {};
}

template <>
llvm::Optional<Value> getAlpha(xten::Conv2dTensorAddReLUOp &reluOp) {
  return {};
}

// fetch input. Specialize if the method is named differently.
template <class Op>
Value getInput(Op op) {
  return *op.getODSOperands(0).begin();
}

inline std::string vector_to_str(std::vector<int64_t> vec_input,
                                 std::string separator = ",") {
  std::string vec_str = "";
  for (auto v : vec_input)
    vec_str += std::to_string(v) + separator;

  if (not vec_str.empty())
    vec_str.pop_back();
  return vec_str;
}

void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
  SmallVector<int64_t, 2> sv;
  if (matchPattern(op, Torch::m_TorchConstantIntList(sv))) {
    for (size_t i = 0; i < sv.size(); i++)
      v.push_back(sv[i]);
  } else if (auto co = op.getDefiningOp<arith::ConstantIntOp>()) {
    v.push_back(co.value());
  }
}

inline std::string typeStr(Torch::BaseTensorType attrType) {
  /* aborts if type is not number type (Int or Float) */
  if (attrType.hasDtype()) {
    auto dtype = attrType.getOptionalDtype();
    if (IntegerType type = dtype.dyn_cast<IntegerType>()) {
      return "int" + std::to_string(type.getWidth());
    } else if (FloatType type = dtype.dyn_cast<FloatType>()) {
      return "float" + std::to_string(type.getWidth());
    }
  }
  return "";
}

#define BYTE_SIZE_IN_BIT 8
inline uint64_t total_bytes(Torch::BaseTensorType attrType,
                            uint64_t total_inputs) {
  /* aborts if type is not number type (Int or Float) */
  uint64_t bit_width = 0;
  if (attrType.hasDtype()) {
    auto dtype = attrType.getOptionalDtype();
    if (IntegerType type = dtype.dyn_cast<IntegerType>()) {
      bit_width = type.getWidth();
    } else if (FloatType type = dtype.dyn_cast<FloatType>()) {
      bit_width = type.getWidth();
    }
  }
  return (bit_width / BYTE_SIZE_IN_BIT) * total_inputs;
}

uint64_t numBytes(const mlir::Type &ty) {
  Torch::BaseTensorType torchTy = ty.cast<Torch::BaseTensorType>();
  uint64_t volume = xilinx::xten::getTensorVolume(torchTy);
  return total_bytes(torchTy, volume);
}

uint64_t storageOfInputAndOutput(Value input, Value output) {
  return numBytes(input.getType()) + numBytes(output.getType());
}

std::string sizesToString(Torch::BaseTensorType &torchTy) {
  std::string shape = "";
  for (auto &d : torchTy.getSizes()) {
    shape += std::to_string(d) + "x";
  }
  shape.pop_back();
  return shape;
}

/// Properties for a particular op
class JsonPropertiesBuilder {
private:
  llvm::json::Array &propertiesArray;
  int numSubOps = 0;
  int unfusedOpId;
  /// type of the unfused op. May be null.
  const char *opTypeStr;

public:
  JsonPropertiesBuilder(llvm::json::Array &propertiesArray)
      : propertiesArray(propertiesArray), unfusedOpId(-1), opTypeStr(nullptr) {}

  JsonPropertiesBuilder(llvm::json::Array &propertiesArray,
                        const char *opTypeStr, int unfusedOpId)
      : propertiesArray(propertiesArray), unfusedOpId(unfusedOpId),
        opTypeStr(opTypeStr) {}

  // append a single property
  void append(std::string name, std::string value) {
    llvm::json::Object propertyObject;
    propertyObject["name"] = name;
    propertyObject["value"] = value;
    if (isFused()) {
      propertyObject["unfused_operator_type"] = opTypeStr;
      propertyObject["unfused_operator_id"] = std::to_string(unfusedOpId);
    }
    // TODO sort by name to make ordering independent of code.
    propertiesArray.push_back(llvm::json::Value(std::move(propertyObject)));
  }

  int64_t appendTypeInfo(const std::string &prefix, const mlir::Type &ty) {

    if (ty.isa<Torch::NoneType>()) {
      append(prefix + ".Tensor", "");
      append(prefix + ".type", "None");
      append(prefix + ".Bytes", "0");
      return 0;
    }

    Torch::BaseTensorType torchTy = ty.cast<Torch::BaseTensorType>();

    uint64_t volume = xilinx::xten::getTensorVolume(torchTy);
    uint64_t bytes = total_bytes(torchTy, volume);

    append(prefix + ".Tensor", sizesToString(torchTy));
    append(prefix + ".type", typeStr(torchTy));
    append(prefix + ".Bytes", std::to_string(bytes));
    return bytes;
  }

  void appendFloatValue(std::string name, const mlir::Value &value) {
    auto valueOp = value.getDefiningOp<Torch::ConstantFloatOp>();
    auto valueStr = std::to_string(valueOp.value().convertToDouble());
    append(name, valueStr);
  }

  void appendIntValue(std::string name, const mlir::Value &value) {
    auto valueOp = value.getDefiningOp<arith::ConstantIntOp>();
    auto valueStr = std::to_string(valueOp.value());
    append(name, valueStr);
  }

  void appendIntList(std::string name, const mlir::Value &value) {
    std::vector<int64_t> vec;
    unpack_int_list(value, vec);
    append(name, vector_to_str(vec));
  }

  void appendBytesAttr(int64_t bytes) {
    append("Storage.Bytes", std::to_string(bytes));
  }

  JsonPropertiesBuilder nextFusedOp(const char *typeStr) {
    return JsonPropertiesBuilder(propertiesArray, typeStr, numSubOps++);
  }

  bool isFused() {
    return unfusedOpId >= 0;
  }

  template <class Op>
  void appendStorageAttr(Op &op, uint64_t bytes) {
    if (!isFused()) {
      Value input = getInput(op);
      Value output = ((Operation *)op)->getResult(0);
      uint64_t storageIoBytes = storageOfInputAndOutput(input, output);
      appendBytesAttr(bytes + storageIoBytes);
    }
  }
};

struct ATenVisualGraphPass : public ATenVisualGraphBase<ATenVisualGraphPass> {

private:
  std::string o;
  std::string &output;

  llvm::MapVector<Operation *, std::pair<std::string, std::string>> opToName;
  std::map<Operation *, int> opToId;

  std::unordered_map<std::string, int> opTypes;
  std::unordered_map<std::string, llvm::json::Object> opTypeToProperties;

  std::map<std::string, std::vector<Operation *>> layerToOps;
  std::unordered_map<int, std::vector<std::string>> designToLayers;

  // map of ops whose inputs come from other ops
  llvm::MapVector<Operation *, llvm::MapVector<Operation *, unsigned>>
      connsInToOutMap;

  // map of ops whose output are sent to other ops
  llvm::MapVector<Operation *, llvm::MapVector<Operation *, unsigned>>
      connsOutToInMap;

  // maps to be defined at the beginning from JSON input model
  std::map<std::string, std::vector<std::vector<std::string>>> propertiesInfo;
  std::unordered_map<std::string, std::vector<std::string>>
      fusedOpToUnfusedOpsMap;

  std::unordered_map<Operation *, std::unordered_map<std::string, std::string>>
      fusedToUnfuseOutputMap;
  std::unordered_map<Operation *, std::unordered_map<std::string, std::string>>
      fusedToUnfuseInputMap;

  std::unordered_map<Operation *, Value> inputOpToIFMValue;


  unsigned currentDesign = 1;

  void initProperties() {
    assert(ATenOperatorsSupportedFilePath != "-" &&
           "Provide (absolute or relative) path to JSON list of operators "
           "supported (operators_supported.json)");

    auto ss = std::ostringstream{};
    std::ifstream file(ATenOperatorsSupportedFilePath);

    ss << file.rdbuf();
    auto json_model_str = std::string(ss.str());
    StringRef sr(json_model_str);

    auto jsonModel = llvm::json::parse(sr);
    assert(jsonModel);
    auto operators = jsonModel->getAsObject();
    if (!operators)
      return;

    auto opArray = operators->getArray("ops");
    if (!opArray)
      return;

    for (auto op : *opArray) {
      auto opObject = op.getAsObject();
      if (!opObject)
        return;
      std::string name_op = opObject->getString("name")->str();
      propertiesInfo[name_op] = std::vector<std::vector<std::string>>();
      auto propertiesArray = opObject->getArray("properties");

      /* format -->   { 'name', 'tooltip', 'type' } */
      for (auto prop : *propertiesArray) {
        auto propVector = std::vector<std::string>(3, "");
        auto propArray = *prop.getAsArray();
        propVector[0] = propArray[0].getAsString()->str();
        propVector[1] = propArray[1].getAsString()->str();
        propVector[2] = propArray[2].getAsString()->str();
        propertiesInfo[name_op].push_back(propVector);
      }

      auto fusedArray = opObject->getArray("fused");
      if (fusedArray) {
        for (auto unfused_op : *fusedArray)
          fusedOpToUnfusedOpsMap[name_op].push_back(
              unfused_op.getAsString()->str());
      }
    }
  }

  inline void clearAllDataStructures() {
    opToName.clear();
    opToId.clear();
    opTypes.clear();
    layerToOps.clear();
    designToLayers.clear();
    connsInToOutMap.clear();
    connsOutToInMap.clear();
    fusedToUnfuseOutputMap.clear();
    inputOpToIFMValue.clear();

  }

  std::map<std::string, uint64_t> getLayerStatsMap(Operation *op) {
    std::map<std::string, uint64_t> layerStatsMap;

    // ------ NOTE: Some functions in getATenOpStats don't handle 'unknown'
    // tensor sizes.------
    // ------ For now, comment out this section till the above is fixed.
    // ----------------------
    /* if (auto stats = mlir::dyn_cast<NPCOMP::StatisticsOpInterface>(op)) {
      layerStatsMap = stats.getStatistics();
    } else {
    layerStatsMap = xilinx::xten::getATenOpStats(op);

    //couldn't find ATen Stats for op, check if it has XTen Stats instead
    if (layerStatsMap.empty())
      layerStatsMap = xilinx::xten::getXTenOpStats(op);
    */

    return layerStatsMap;
  }

  inline std::string getOperationNameStr(Operation *op) {
    return op->getName().getStringRef().str();
  }

  inline void updateOperatorTypes(Operation *op) {
    auto op_str = getOperationNameStr(op);
    opTypes[op_str]++;
  }

  inline bool opIsValid(Operation *op) {
    if (!op)
      return false;

    return propertiesInfo.find(getOperationNameStr(op)) != propertiesInfo.end();
  }

  template <class T>
  void fillPropertiesUnaryALUOp(T &op, JsonPropertiesBuilder &&props) {
    props.appendStorageAttr(op, 0);
  }

  template <class T>
  void fillOther(T &op, Value v, JsonPropertiesBuilder &props) {
    auto volume_bytes =
        props.appendTypeInfo("Attributes.Other", v.getType());
    props.appendStorageAttr(op, volume_bytes);
  }

  template <class T>
  void fillPropertiesBinaryALUOp(T &op, JsonPropertiesBuilder &&props) {
    fillOther(op, op.other(), props);
  }

  void fillProperties(Torch::AtenMmOp &op, JsonPropertiesBuilder &&props) {
    fillOther(op, op.mat2(), props);
  }

  void fillProperties(xten::MMOp &op, JsonPropertiesBuilder &&props) {
    fillOther(op, op.getY(), props);
  }

  void fillProperties(xten::AddOp &op, JsonPropertiesBuilder &&props) {
    fillOther(op, op.getInput1(), props);
  }

  void fillPropertiesCatOp(Torch::AtenCatOp &op,
                           JsonPropertiesBuilder &&props) {
    props.appendStorageAttr(op, 0);

    //    std::string storage_str = ""; // what?
    //  fillPropertiesObject({"Storage.Bytes", storage_str}, propertiesArray);
  }

  template <class ConvOp>
  inline Value getWeight(ConvOp op) {
    return op.getWeight();
  }

  template <>
  inline Value getWeight(Torch::AtenConvolutionOp op) {
    return op.weight();
  }

  template <class ConvOp>
  inline Value getBias(ConvOp op) {
    return op.getBias();
  }

  template <>
  inline Value getBias(Torch::AtenConvolutionOp op) {
    return op.bias();
  }

  template <class ConvOp>
  inline Value getConvPadding(ConvOp op) {
    return op.getPadding();
  }

  template <>
  inline Value getConvPadding(Torch::AtenConvolutionOp op) {
    return op.padding();
  }

  template <class ConvOp>
  inline Value getConvStride(ConvOp op) {
    return op.getStride();
  }

  template <>
  inline Value getConvStride(Torch::AtenConvolutionOp op) {
    return op.stride();
  }

  template <class ConvOp>
  inline Value getDilation(ConvOp op) {
    return op.getDilation();
  }

  template <>
  inline Value getDilation(Torch::AtenConvolutionOp op) {
    return op.dilation();
  }

  template <class T>
  uint64_t fillPropertiesConvOp(T &op, JsonPropertiesBuilder &&props) {
    // note that the shape originally was written out as ?o?c?h?w,
    // now it's ?x?x?x? like everywhere else.
    auto bytes = 0;
    bytes += props.appendTypeInfo("Attributes.Weights", getWeight(op).getType());
    bytes += props.appendTypeInfo("Attributes.Bias", getBias(op).getType());


    Torch::BaseTensorType weightTy =
        getWeight(op).getType().template cast<Torch::BaseTensorType>();

    // h,w
    std::string kernel_shape =
        std::to_string( weightTy.getSizes()[2]) 
        + "," + std::to_string( weightTy.getSizes()[3]);

    props.append("Attributes.kernel shape", kernel_shape);
    props.appendIntList("Attributes.padding", getConvPadding(op));
    props.appendIntList("Attributes.stride", getConvStride(op));
    props.appendIntList("Attributes.dilation", getDilation(op));

    std::map<std::string, uint64_t> layerStatsMap = getLayerStatsMap(op);
    props.append("Computations.MAC", std::to_string(layerStatsMap["ops:MAC"]));

    props.appendStorageAttr(op, bytes);
    return bytes;
  }

  template <class MaxpoolOp>
  Value getKernelSize(MaxpoolOp &maxPoolOp) {
    return maxPoolOp.getMpKernelSize();
  }

  template <class MaxpoolOp>
  Value getStride(MaxpoolOp &maxPoolOp) {
    return maxPoolOp.getMpStride();
  }

  template <class MaxpoolOp>
  Value getPadding(MaxpoolOp &maxPoolOp) {
    return maxPoolOp.getMpPadding();
  }

  template <>
  Value getKernelSize(Torch::AtenMaxPool2dOp &maxPoolOp) {
    return maxPoolOp.kernel_size();
  }

  template <>
  Value getStride(Torch::AtenMaxPool2dOp &maxPoolOp) {
    return maxPoolOp.stride();
  }

  template <>
  Value getPadding(Torch::AtenMaxPool2dOp &maxPoolOp) {
    return maxPoolOp.padding();
  }

  template <class T>
  uint64_t fillPropertiesMaxPool2dOp(T &op, JsonPropertiesBuilder &&props) {

    uint64_t bytes =
        0; // FIXME is always zero, was already the case before refactoring

    props.appendIntList("Attributes.kernel shape", getKernelSize(op));
    props.appendIntList("Attributes.padding", getPadding(op));
    props.appendIntList("Attributes.stride", getStride(op));

    std::map<std::string, uint64_t> layerStatsMap = getLayerStatsMap(op);
    props.append("Computations.Vec MAX",
                 std::to_string(layerStatsMap["ops:>"]));

    props.appendStorageAttr(op, bytes);
    return bytes;
  }

  template <class T>
  uint64_t fillPropertiesReLUOp(T &op, JsonPropertiesBuilder &&props) {
    uint64_t bytes = 0;
    auto alphaOpt = getAlpha(op);
    if (alphaOpt) {
      Value alpha = *alphaOpt;
      std::string alpha_str;
      std::string alpha_type_str;
      uint64_t alpha_bytes = 0;

      if (auto co = alpha.getDefiningOp<Torch::ConstantFloatOp>()) {
        alpha_str = std::to_string(co.value().convertToDouble());
        alpha_bytes = sizeof(double);
        alpha_type_str = "float" + std::to_string(alpha_bytes);
      } else {
        int64_t val = alpha.getDefiningOp<arith::ConstantIntOp>().value();
        alpha_str = std::to_string(val);
        alpha_bytes = alpha.getType().dyn_cast<const IntegerType>().getWidth();
        alpha_type_str = "int" + std::to_string(alpha_bytes);
      }
      alpha_bytes /= BYTE_SIZE_IN_BIT;

      props.append("Attributes.Alpha.Tensor", "1");
      props.append("Attributes.Alpha.type", alpha_type_str);
      props.append("Attributes.Alpha.Bytes", std::to_string(alpha_bytes));
      props.append("Attributes.Alpha", alpha_str);
      bytes += alpha_bytes;
    }

    std::map<std::string, uint64_t> layerStatsMap = getLayerStatsMap(op);
    props.append("Computations.Comparison",
                 std::to_string(layerStatsMap["ops:>"]));

    props.appendStorageAttr(op, bytes);
    return bytes;
  }

  inline Value getWeight(Torch::AtenLinearOp linOp) {
    return linOp.weight();
  }

  inline Value getWeight(xten::LinearOp xtenLinOp) {
    return xtenLinOp.getWeight();
  }

  inline Value getBias(Torch::AtenLinearOp linOp) {
    return linOp.bias();
  }

  inline Value getBias(xten::LinearOp xtenLinOp) {
    return xtenLinOp.getBias();
  }

  template <typename LinearOpType>
  void fillPropertiesLinearOp(LinearOpType &op, JsonPropertiesBuilder &&props) {
    auto bytes = 0;
    bytes += props.appendTypeInfo("Attributes.Weights", getWeight(op).getType());
    bytes += props.appendTypeInfo("Attributes.Bias", getBias(op).getType());
    props.appendStorageAttr(op, bytes);
  }

  inline Value getBnWeight(Torch::AtenBatchNormOp bnOp) {
    return bnOp.weight();
  }

  inline Value getBnWeight(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getBnWeight();
  }

  inline Value getBnBias(Torch::AtenBatchNormOp bnOp) {
    return bnOp.bias();
  }

  inline Value getBnBias(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getBnBias();
  }

  inline Value getRunningMean(Torch::AtenBatchNormOp bnOp) {
    return bnOp.running_mean();
  }

  inline Value getRunningMean(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getRunningMean();
  }

  inline Value getRunningVar(Torch::AtenBatchNormOp bnOp) {
    return bnOp.running_var();
  }

  inline Value getRunningVar(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getRunningVar();
  }

  inline Value getEps(Torch::AtenBatchNormOp bnOp) {
    return bnOp.eps();
  }

  inline Value getEps(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getEps();
  }

  inline Value getMomentum(Torch::AtenBatchNormOp bnOp) {
    return bnOp.momentum();
  }

  inline Value getMomentum(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.getMomentum();
  }

  template <class T>
  uint64_t fillPropertiesBatchNormOp(T &op, JsonPropertiesBuilder &&props) {
    uint64_t bytes = 0;
    bytes +=
        props.appendTypeInfo("Attributes.Weights", getBnWeight(op).getType());
    bytes += props.appendTypeInfo("Attributes.Bias", getBnBias(op).getType());
    bytes +=
        props.appendTypeInfo("Attributes.Weights", getRunningMean(op).getType());
    bytes +=
        props.appendTypeInfo("Attributes.Variance", getRunningVar(op).getType());
    props.appendStorageAttr(op, bytes);

    props.appendFloatValue("Attributes.eps", getEps(op));
    props.appendFloatValue("Attributes.momentum", getMomentum(op));
    return bytes;
  }

  template <class T>
  void fillPropertiesSoftmaxOp(T &op, JsonPropertiesBuilder &&props) {
    props.appendIntValue("Attributes.dim", op.dim());
    props.appendStorageAttr(op, 0);
  }

  void fillPropertiesGatherOp(Torch::AtenGatherOp &op,
                              JsonPropertiesBuilder &&props) {
    props.appendIntValue("Attributes.dim", op.dim());
    props.appendStorageAttr(op, 0);
    props.appendTypeInfo("Attributes.Index", op.index().getType());
  }

  void fillPropertiesSliceOp(Torch::AtenSliceTensorOp &op,
                             JsonPropertiesBuilder &&props) {
    props.appendIntValue("Attributes.dim", op.dim());
    props.appendStorageAttr(op, 0);
  }

  void fillPropertiesOp(xten::Conv2dBatchNormReLUOp &op,
                        JsonPropertiesBuilder &&props) {
    uint64_t storage = 0;
    storage += fillPropertiesConvOp<xten::Conv2dBatchNormReLUOp>(
        op, props.nextFusedOp("aten.convolution"));
    storage += fillPropertiesBatchNormOp<xten::Conv2dBatchNormReLUOp>(
        op, props.nextFusedOp("torch.aten.batch_norm"));


    props.appendStorageAttr(op, storage);
  }

  template <class Op>
  void fillPropertiesOpC2dActMaxpool(Op &op, const char *actTypeStr,
                                     JsonPropertiesBuilder &&props) {
    uint64_t storage = 0;
    storage += fillPropertiesConvOp(op, props.nextFusedOp("aten.convolution"));
    storage += fillPropertiesReLUOp(op, props.nextFusedOp(actTypeStr));
    storage +=
        fillPropertiesMaxPool2dOp(op, props.nextFusedOp("aten.max_pool2d"));

    props.appendStorageAttr(op, storage);
  }

  template <class Op>
  void fillPropertiesOpC2dAct(Op &op, const char *actTypeStr,
                              JsonPropertiesBuilder &&props) {
    uint64_t storage = 0;
    storage += fillPropertiesConvOp(op, props.nextFusedOp("aten.convolution"));
    storage += fillPropertiesReLUOp(op, props.nextFusedOp(actTypeStr));
    props.appendStorageAttr(op, storage);
  }

  llvm::json::Object emitJSONSchema() {
    llvm::json::Object schemaObject;

    schemaObject["major"] = "1";
    schemaObject["minor"] = "0";
    schemaObject["patch"] = "0";

    return schemaObject;
  }

  llvm::json::Array emitJSONOperators() {
    llvm::json::Array operatorsArray;

    for (auto const &op_type_pair : opTypes) {
      llvm::json::Object operatorObject;

      auto op_str = op_type_pair.first;
      operatorObject["name"] = op_str;
      operatorObject["description"] = "ATen/XTen operator " + op_str;

      llvm::json::Array operatorPropsArray;
      for (auto const &op_props_vec : propertiesInfo[op_str]) {
        llvm::json::Object operatorPropsObject;
        operatorPropsObject["name"] = op_props_vec[0];
        operatorPropsObject["tooltip"] = op_props_vec[1];
        operatorPropsObject["type"] = op_props_vec[2];

        operatorPropsArray.push_back(
            llvm::json::Value(std::move(operatorPropsObject)));
      }

      operatorObject["properties"] =
          llvm::json::Value(std::move(operatorPropsArray));
      operatorsArray.push_back(llvm::json::Value(std::move(operatorObject)));
    }

    return operatorsArray;
  }

  llvm::json::Array emitJSONDesigns() {
    llvm::json::Array designsArray;

    for (auto const &design_pair : designToLayers) {
      llvm::json::Object designObject;
      llvm::json::Object propertyObject;
      llvm::json::Array propertyArray;

      int design_id = design_pair.first;
      int layer_count = design_pair.second.size();

      propertyObject["name"] = "Layer Count";
      propertyObject["tooltip"] = "Total number of layers in design";
      propertyObject["type"] = "int";
      propertyObject["value"] = std::to_string(layer_count);
      propertyArray.push_back(llvm::json::Value(std::move(propertyObject)));

      designObject["name"] = "design " + std::to_string(design_id);
      designObject["properties"] = llvm::json::Value(std::move(propertyArray));

      designsArray.push_back(llvm::json::Value(std::move(designObject)));
    }

    return designsArray;
  }

  llvm::json::Array fillProperties(Operation *op) {
    llvm::json::Array propertiesArray;
    auto props = JsonPropertiesBuilder(propertiesArray);
    if (auto op2 = dyn_cast<Torch::AtenConvolutionOp>(op)) {
      fillPropertiesConvOp<Torch::AtenConvolutionOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenMaxPool2dOp>(op)) {
      fillPropertiesMaxPool2dOp<Torch::AtenMaxPool2dOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenReluOp>(op)) {
      fillPropertiesReLUOp<Torch::AtenReluOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenBatchNormOp>(op)) {
      fillPropertiesBatchNormOp<Torch::AtenBatchNormOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenLinearOp>(op)) {
      fillPropertiesLinearOp(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenSizeOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSizeOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenCatOp>(op)) {
      fillPropertiesCatOp(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenNegOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenNegOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenSigmoidOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSigmoidOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenSinOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSinOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenTanhOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenTanhOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenExpOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenExpOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenAddTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenAddTensorOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenMulTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenMulTensorOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenDivTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenDivTensorOp>(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenGatherOp>(op)) {
      fillPropertiesGatherOp(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenSliceTensorOp>(op)) {
      fillPropertiesSliceOp(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenConstantPadNdOp>(op)) {
      props.appendIntList("Attributes.padding", op2.pad());
    } else if (auto op2 = dyn_cast<Torch::AtenMeanDimOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 = dyn_cast<Torch::AtenSqueezeDimOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 = dyn_cast<Torch::AtenMmOp>(op)) {
      fillProperties(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::Aten_SoftmaxOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 = dyn_cast<xten::GlobalAveragePool2D>(op)) {
      // TODO: fill properties
    } else if (auto op2 = dyn_cast<xten::LinearOp>(op)) {
      fillPropertiesLinearOp(op2, std::move(props));
    } else if (auto op2 = dyn_cast<Torch::AtenArgmaxOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dOp>(op)) {
      fillPropertiesConvOp<xten::Conv2dOp>(op2, std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
      fillPropertiesOp(op2, std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
      fillPropertiesOpC2dAct(op2, "aten.lrelu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUOp>(op)) {
      fillPropertiesOpC2dAct(op2, "aten.relu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      fillPropertiesOpC2dActMaxpool(op2, "aten.lrelu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUMaxPoolOp>(op)) {
      fillPropertiesOpC2dActMaxpool(op2, "aten.relu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dTensorAddReLUOp>(op)) {
      fillPropertiesOpC2dAct(op2, "aten.relu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::AddOp>(op)) {
      fillProperties(op2, std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::MMOp>(op)) {
      fillProperties(op2, std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUPadMaxPoolOp>(op)) {
      // todo pad attributes are missing
      fillPropertiesOpC2dActMaxpool(op2, "aten.lrelu", std::move(props));
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUPadMaxPoolOp>(op)) {
      // todo pad attributes are missing
      fillPropertiesOpC2dActMaxpool(op2, "aten.relu", std::move(props));
    } else if (auto op2 =
                   dyn_cast<xten::Conv2dTensorAddGlobalAveragePoolOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 =
                   dyn_cast<xten::Conv2dTensorAddReLUGlobalAveragePoolOp>(op)) {
      // TODO: fill properties
    } else if (auto op2 =
                   dyn_cast<xten::Conv2dTensorAddLReLUGlobalAveragePoolOp>(
                       op)) {
      // TODO: fill properties
    }

    return propertiesArray;
  }

  Value getInputFromErasedPtr(Operation *op) {
    Value opInput;

    if (auto op2 = dyn_cast<Torch::AtenConvolutionOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = dyn_cast<Torch::AtenMaxPool2dOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = dyn_cast<Torch::AtenReluOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = dyn_cast<Torch::AtenAddTensorOp>(op)) {
      opInput = getInput(op2); // TODO: add Tensor technically has two inputs
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dLReLUPadMaxPoolOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUMaxPoolOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<xten::Conv2dReLUPadMaxPoolOp>(op)) {
      opInput = getInput(op2);
    } else if (auto op2 = mlir::dyn_cast<Torch::AtenFlattenUsingIntsOp>(op)) {
      opInput = getInput(op2);
    } else {
      llvm_unreachable("Unhandled op");
    }
    // TODO: expand switch table for more ops

    return opInput;
  }

  void fillPortProperties(Operation *op, bool isInput,
                          llvm::json::Array &portPropsArray,
                          bool unfused_part_of_fused = false,
                          std::string unfused_part_name = "",
                          unsigned unfused_id = 0) {
    std::string port_name_prefix;
    std::string port_type_str;
    if (isInput) {
      port_name_prefix = "Inputs.IFMs";
      port_type_str = "Input";
    } else {
      port_name_prefix = "Outputs.OFMs";
      port_type_str = "Output";
    }

    llvm::json::Object portPropsObject;
    if (unfused_part_of_fused) {
      portPropsObject["name"] = port_name_prefix + ".Tensor";
      portPropsObject["tooltip"] = "Dimensions of " + port_type_str;
      portPropsObject["type"] = "string";

      /* fusedToUnfuseOutputMaps should have been filled by previous
       * fillProperties functions */
      portPropsObject["value"] = fusedToUnfuseOutputMap[op][unfused_part_name];

      portPropsObject["unfused_operator_type"] = unfused_part_name;
      portPropsObject["unfused_operator_id"] = std::to_string(unfused_id);
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));

    } else {
      std::string value_str = "";
      std::string type_str = "";
      std::string bytes_str = "";
      uint64_t total_inputs = 0;

      // TODO: for operators with multiple outputs, will need to go over them
      Type resultTy =
          (inputOpToIFMValue.find(op) != inputOpToIFMValue.end() and isInput
               ? inputOpToIFMValue[op]
               : op->getResult(0))
              .getType();
      Torch::BaseTensorType sizeResultTy =
          resultTy.dyn_cast<Torch::BaseTensorType>();
      if (!sizeResultTy)
        return;

      if (sizeResultTy.hasSizes()) {
        total_inputs = xilinx::xten::getTensorVolume(sizeResultTy);
        value_str = sizesToString(sizeResultTy);
        type_str = typeStr(sizeResultTy);
        bytes_str = std::to_string(total_bytes(sizeResultTy, total_inputs));
      }


      portPropsObject["name"] = port_name_prefix + ".Tensor";
      portPropsObject["tooltip"] = "Dimensions of " + port_type_str;
      portPropsObject["type"] = "string";
      portPropsObject["value"] = value_str;
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));

      portPropsObject["name"] = port_name_prefix + ".type";
      portPropsObject["tooltip"] =
          "Numerical type of each element of " + port_type_str;
      portPropsObject["type"] = "string";
      portPropsObject["value"] = type_str;
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));

      portPropsObject["name"] = port_name_prefix + ".Bytes";
      portPropsObject["tooltip"] = "Size of " + port_type_str + " in bytes";
      portPropsObject["type"] = "long";
      portPropsObject["value"] = bytes_str;
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));
    }
  }

  llvm::json::Array emitJSONLayerOpPorts(Operation *op) {
    llvm::json::Array portsArray;

    unsigned input_port_id = 0;
    for (auto const &op_input : connsInToOutMap[op]) {
      llvm::json::Object portObject;

      portObject["id"] = std::to_string(op_input.second);
      portObject["name"] = "in_" + std::to_string(input_port_id++);
      portObject["direction"] = "in";

      llvm::json::Array portPropsArray;

      fillPortProperties(op_input.first, true, portPropsArray);

      portObject["properties"] = llvm::json::Value(std::move(portPropsArray));
      portsArray.push_back(llvm::json::Value(std::move(portObject)));
    }

    unsigned output_port_id = 0;
    for (auto const &op_output : connsOutToInMap[op]) {
      llvm::json::Object portObject;

      portObject["id"] = std::to_string(op_output.second);
      portObject["name"] = "out_" + std::to_string(output_port_id++);
      portObject["direction"] = "out";

      llvm::json::Array portPropsArray;
      fillPortProperties(op, false, portPropsArray);

      /*For each fused operator, also add to JSON model sub-outputs of unfused
       * parts of fused op  */
      auto op_name = getOperationNameStr(op);
      if (fusedOpToUnfusedOpsMap.find(op_name) !=
          fusedOpToUnfusedOpsMap.end()) {
        unsigned unfused_id = 1;
        for (auto unfused_op_name : fusedOpToUnfusedOpsMap[op_name]) {
          fillPortProperties(op, false, portPropsArray, true, unfused_op_name,
                             unfused_id++);
        }
      }

      portObject["properties"] = llvm::json::Value(std::move(portPropsArray));
      portsArray.push_back(llvm::json::Value(std::move(portObject)));
    }

    return portsArray;
  }

  llvm::json::Array emitJSONLayers() {
    llvm::json::Array layersArray;

    for (auto const &layer_pair : layerToOps) {
      llvm::json::Object layerObject;
      llvm::json::Object propertyObject;
      llvm::json::Array propertiesArray;
      llvm::json::Array operatorsArray;

      auto layer_name = layer_pair.first;
      auto layer_ops = layer_pair.second;
      int layer_ops_count = layer_ops.size();

      propertyObject["name"] = "Operator Count";
      propertyObject["tooltip"] =
          "Total number of ATen operators in layer to be visualized";
      propertyObject["type"] = "int";
      propertyObject["value"] = std::to_string(layer_ops_count);

      layerObject["name"] = layer_name;
      layerObject["design_name"] = "design " + std::to_string(currentDesign);

      propertiesArray.push_back(llvm::json::Value(std::move(propertyObject)));
      layerObject["properties"] = llvm::json::Value(std::move(propertiesArray));
      for (auto op : layer_ops) {
        llvm::json::Object operatorObject;
        auto op_id_str = opToName[op].first;

        operatorObject["id"] = std::to_string(opToId[op]);
        operatorObject["name"] = op_id_str;
        operatorObject["operator_type"] = getOperationNameStr(op);

        auto propertiesArray = fillProperties(op);
        operatorObject["properties"] =
            llvm::json::Value(std::move(propertiesArray));

        auto portsArray = emitJSONLayerOpPorts(op);
        operatorObject["ports"] = llvm::json::Value(std::move(portsArray));

        operatorsArray.push_back(llvm::json::Value(std::move(operatorObject)));
      }

      layerObject["operators"] = llvm::json::Value(std::move(operatorsArray));
      layersArray.push_back(llvm::json::Value(std::move(layerObject)));
    }

    return layersArray;
  }

  llvm::json::Array emitJSONConnections() {
    llvm::json::Array connectionsArray;

    unsigned connection_id = 0;
    for (auto const &conns_pair : connsInToOutMap) {
      Operation *input_op = conns_pair.first;
      const auto output_ops = conns_pair.second;
      for (auto const &output_op_pair : output_ops) {
        Operation *output_op = output_op_pair.first;
        if (output_op == input_op)
          continue;

        llvm::json::Object connectionObject;

        unsigned inPortId = output_op_pair.second;
        unsigned outPortId = connsOutToInMap[output_op][input_op];

        connectionObject["id"] = std::to_string(connection_id++);
        connectionObject["from_port_id"] = std::to_string(outPortId);
        connectionObject["to_port_id"] = std::to_string(inPortId);
        connectionsArray.push_back(
            llvm::json::Value(std::move(connectionObject)));
      }
    }

    return connectionsArray;
  }

  std::vector<Operation *> vectorValidArgOps(Operation *argOp, Operation *op) {
    auto vectorArgOps = std::vector<Operation *>();
    if (opIsValid(argOp))
      vectorArgOps.push_back(argOp);
    else if (getOperationNameStr(op) == "torch.aten.cat" and
             getOperationNameStr(argOp) == "torch.prim.ListConstruct") {
      for (auto listArgOp_ref : argOp->getOperands()) {
        if (auto listArgOp = listArgOp_ref.getDefiningOp())
          if (opIsValid(listArgOp))
            vectorArgOps.push_back(listArgOp);
      }
    }
    return vectorArgOps;
  }

public:
  Option<std::string> ATenVisualGraphFilename{
      *this, "output-file", llvm::cl::desc("Output filename for JSON report"),
      llvm::cl::init("-")};

  Option<std::string> ATenOperatorsSupportedFilePath{
      *this, "operators-supported-path",
      llvm::cl::desc(
          "Path of JSON file that has list of operators supported (REQUIRED)"),
      llvm::cl::init("-")};

  ATenVisualGraphPass(const ATenVisualGraphPass &pass) : output(o) {}

  ATenVisualGraphPass() : output(o) {}

  std::string emitJSONReport() {
    llvm::json::Object top;

    llvm::json::Object flexmlTop;
    llvm::json::Object flexmlSchema;
    llvm::json::Array flexmlDesigns;
    llvm::json::Array flexmlOperators;
    llvm::json::Array flexmlLayers;
    llvm::json::Array flexmlConnections;

    flexmlSchema = emitJSONSchema();
    flexmlOperators = emitJSONOperators();
    flexmlDesigns = emitJSONDesigns();
    flexmlLayers = emitJSONLayers();
    flexmlConnections = emitJSONConnections();

    // Fill Top JSON properties
    flexmlTop["schema_version"] = llvm::json::Value(std::move(flexmlSchema));
    flexmlTop["designs"] = llvm::json::Value(std::move(flexmlDesigns));
    flexmlTop["operator_types"] = llvm::json::Value(std::move(flexmlOperators));
    flexmlTop["layers"] = llvm::json::Value(std::move(flexmlLayers));
    flexmlTop["connections"] = llvm::json::Value(std::move(flexmlConnections));

    std::string topName = "flexml_graph_metadata";
    top[topName] = llvm::json::Value(std::move(flexmlTop));

    llvm::json::Value topv(std::move(top));
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << llvm::formatv("{0:2}", topv) << "\n";
    return ss.str();
  }

  void runOnOperation() override {
    initProperties();

    // I don't change anything
    markAllAnalysesPreserved();

    auto module = getOperation();

    // check that a function called "forward" exists
    auto forward = module.lookupSymbol<func::FuncOp>("forward");
    if (!forward) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a forward function\n");
      signalPassFailure();
      return;
    }

    clearAllDataStructures();

    unsigned currentOp = 0;
    unsigned currPortId = 0;
    forward.walk([&](Operation *op) {
      if (!opIsValid(op))
        return;

      auto attr_l = op->getAttrOfType<StringAttr>("layer_name");
      auto attr_n = op->getAttrOfType<StringAttr>("name");
      // assumes layer_name is given to all nodes, might support inferring
      // layers later
      if (!attr_l && !attr_n)
        return;
      auto attr = attr_l ? attr_l : attr_n;

      auto op_str = getOperationNameStr(op);
      auto layer_name = attr.getValue().str();
      layerToOps[layer_name].push_back(op);
      designToLayers[currentDesign].push_back(layer_name);

      opToName[op] = std::make_pair(
          op_str + "_" + std::to_string(opTypes[op_str]), layer_name);
      opToId[op] = currentOp;

      for (auto argOp_ref : op->getOperands()) {
        if (auto argOp = argOp_ref.getDefiningOp()) {
          auto vectorArgOps = vectorValidArgOps(argOp, op);
          for (auto argOp : vectorArgOps) {
            // where Op accepts ArgOp as input  <===> Op(%ArgOp, ...)
            connsInToOutMap[op][argOp] = currPortId++;

            // where output of ArgOp is sent to input port of Op <===>
            // ArgOp.output ----> Op.input_port_example
            // same output port goes to several next input ports -> this output
            // port should have one Port ID
            if (connsOutToInMap[argOp].empty())
              connsOutToInMap[argOp][op] = currPortId++;
            else
              connsOutToInMap[argOp][op] =
                  connsOutToInMap[argOp].begin()->second;
          }
        }
      }

      updateOperatorTypes(op);
      currentOp++;
    });

    /* for operators whose outputs are the return values of 'graph' */
    for (const auto &op_pair : opToName) {
      auto op = op_pair.first;
      if (connsInToOutMap.find(op) == connsInToOutMap.end()) {
        // For the first input Ops (aka source Ops) in the NN graph
        connsInToOutMap[op][op] = currPortId++;
        inputOpToIFMValue[op] = getInputFromErasedPtr(op);
      }
      if (connsOutToInMap.find(op) == connsOutToInMap.end())
        connsOutToInMap[op][op] = currPortId++;
    }

    output = emitJSONReport();

    if (ATenVisualGraphFilename != "-") {
      std::error_code EC;
      llvm::raw_fd_ostream aie_ostream(ATenVisualGraphFilename, EC);
      aie_ostream << output;
    } else {
      llvm::outs() << output;
    }
  }
};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<OperationPass<ModuleOp>> createATenVisualGraphPass() {
  return std::make_unique<ATenVisualGraphPass>();
}

} // namespace xten
} // namespace xilinx
