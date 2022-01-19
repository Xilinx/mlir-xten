//===- ATenVisualGraph.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/MapVector.h"

#include "mlir/Pass/Pass.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "xten/Transform/ATenVisualGraph.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#define DEBUG_TYPE "aten-visual-graph"

using namespace mlir;
using namespace xilinx::xten;
using namespace xilinx;
using namespace mlir::torch;

namespace {
struct ATenVisualGraphPass : public ATenVisualGraphBase<ATenVisualGraphPass> {

private:
  std::string o;
  std::string &output;
  
  llvm::MapVector<Operation *, std::pair<std::string, std::string> > opToName;
  std::map<Operation *, int> opToId;
  
  std::unordered_map<std::string, int> opTypes;
  std::unordered_map<std::string, llvm::json::Object> opTypeToProperties;

  std::map<std::string, std::vector<Operation *>> layerToOps;
  std::unordered_map<int, std::vector<std::string>> designToLayers;
  
  //map of ops whose inputs come from other ops  
  llvm::MapVector<Operation *, llvm::MapVector<Operation *, unsigned>> connsInToOutMap;

  //map of ops whose output are sent to other ops
  llvm::MapVector<Operation *, llvm::MapVector<Operation *, unsigned>> connsOutToInMap;

  //maps to be defined at the beginning from JSON input model
  std::map<std::string, std::vector<std::vector<std::string>>> propertiesInfo; 
  std::unordered_map<std::string, std::vector<std::string>> fusedOpToUnfusedOpsMap;
  
  std::unordered_map<Operation *, std::unordered_map<std::string, std::string>> fusedToUnfuseOutputMap;	   
  std::unordered_map<Operation *, std::unordered_map<std::string, std::string>> fusedToUnfuseInputMap;
  
  std::unordered_map<Operation *, Value > inputOpToIFMValue;

  ///// ------------- NOTE: For TinyYolo DEMO only --------------------------------------------------------
  ///// Since Torch-Mlir doesn't propagate tensor output sizes by default, the ATEN visualizer needs 
  ///// include a way to do shape propagation for now. 
  ///// Code Sections marked with a DEMO label implement the shape propagation logic, which is relevant
  ///// to the TinyYolo DEMO only.
  bool propagate_tensor_sizes;

  std::unordered_map<Operation *,
		     std::unordered_map<bool,
					std::unordered_map<std::string,
							   std::vector<uint64_t> >>> propagatedTensorSizesMap;

  std::unordered_map<Operation *, std::string> propagatedTensorTypeMap;
  ///////////////////////
  
  unsigned currentDesign = 1;
  
  void initProperties() {
    assert(ATenOperatorsSupportedFilePath != "-" && "Provide (absolute or relative) path to JSON list of operators supported (operators_supported.json)");
    
    auto ss = std::ostringstream{};
    std::ifstream file(ATenOperatorsSupportedFilePath);

    ss << file.rdbuf();
    auto json_model_str = std::string(ss.str());
    StringRef sr(json_model_str);

    auto jsonModel = llvm::json::parse(sr);
    assert(jsonModel);
    auto operators = jsonModel->getAsObject();
    if (!operators) return;

    auto opArray = operators->getArray("ops");
    if (!opArray) return;
    
    for (auto op : *opArray) {
      auto opObject = op.getAsObject();
      if (!opObject) return;
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
	  fusedOpToUnfusedOpsMap[name_op].push_back(unfused_op.getAsString()->str());
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

    //For DEMO
    propagatedTensorSizesMap.clear();
    propagatedTensorTypeMap.clear();
  }
  
  inline std::string vector_to_str(std::vector<int64_t> vec_input,
				   std::string separator = ",") {
    std::string vec_str = "";
    for (auto v : vec_input)
      vec_str += std::to_string(v) + separator;

    if (not vec_str.empty()) vec_str.pop_back();
    return vec_str;
  }

  void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
    SmallVector<int64_t, 2> sv;
    if (matchPattern(op, Torch::m_TorchConstantIntList(sv)) ) {
      for (size_t i = 0; i < sv.size(); i++)
	v.push_back(sv[i]);
    }
    else if (auto co = op.getDefiningOp<mlir::arith::ConstantIntOp>()) {
      v.push_back(co.value());
    }
  }

  std::map<std::string, uint64_t> getLayerStatsMap(Operation *op) {
    std::map<std::string, uint64_t> layerStatsMap;

    // ------ NOTE: Some functions in getATenOpStats don't handle 'unknown' tensor sizes.------
    // ------ For now, comment out this section till the above is fixed. ----------------------
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

  inline std::string typeStr(Torch::BaseTensorType attrType) {
    /* aborts if type is not number type (Int or Float) */
    if (attrType.hasDtype()){
      auto dtype = attrType.getOptionalDtype();
      if (IntegerType type = dtype.dyn_cast<IntegerType>()) {
	return "int"   + std::to_string(type.getWidth());      
      } else if (FloatType type = dtype.dyn_cast<FloatType>()) {
	return "float" + std::to_string(type.getWidth());
      }
    }
    return "";
  }
  
  #define BYTE_SIZE_IN_BIT 8
  inline uint64_t total_bytes(Torch::BaseTensorType attrType, uint64_t total_inputs) {
    /* aborts if type is not number type (Int or Float) */
    uint64_t bit_width = 0;
    if (attrType.hasDtype()){
      auto dtype = attrType.getOptionalDtype();
      if (IntegerType type = dtype.dyn_cast<IntegerType>()) {
	bit_width = type.getWidth();      
      } else if (FloatType type = dtype.dyn_cast<FloatType>()) {
	bit_width = type.getWidth();     
      }
    }
    return (bit_width/BYTE_SIZE_IN_BIT) * total_inputs;
  }

  uint64_t storage_bytes_of_input_and_output(Value input, Value output) {
    uint64_t storage_n = 0;

    Torch::BaseTensorType inputTy  = input.getType().cast<Torch::BaseTensorType>();
    Torch::BaseTensorType outputTy = output.getType().cast<Torch::BaseTensorType>();

    uint64_t i_num_inputs = xilinx::xten::getTensorVolume(inputTy);
    uint64_t o_num_inputs = xilinx::xten::getTensorVolume(outputTy);
    
    uint64_t input_bytes   = total_bytes(inputTy, i_num_inputs);
    uint64_t output_bytes  = total_bytes(outputTy, o_num_inputs);

    storage_n = input_bytes + output_bytes;
    return storage_n;
  }
  
  void fillPropertiesObject(std::vector<std::string> properties_vec,
			    llvm::json::Array &propertiesArray,
			    bool is_fused = false, std::string opTypeStr = "",
			    unsigned unfused_op_id = 0) {
      llvm::json::Object propertyObject;
      propertyObject["name"]  = properties_vec[0];
      propertyObject["value"] = properties_vec[1];
      if (is_fused) {
	propertyObject["unfused_operator_type"]  = opTypeStr;
	propertyObject["unfused_operator_id"]    = std::to_string(unfused_op_id);
      }
      propertiesArray.push_back(llvm::json::Value(std::move(propertyObject)) );          
  }

  template<class T>
  void fillPropertiesUnaryALUOp(T &aluOp, llvm::json::Array &propertiesArray) {
    Value input  = aluOp.self();
    Value output = aluOp.getResult();

    uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
    std::string storage_str      = std::to_string(storage_i_o_bytes);

    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);
  }

  template<class T>
  void fillPropertiesBinaryALUOp(T &aluOp, llvm::json::Array &propertiesArray) {
    Value input  = aluOp.self();
    Value other  = aluOp.other();
    Value output = aluOp.getResult();

    Torch::BaseTensorType otherTy = other.getType().cast<Torch::BaseTensorType>();
    std::string other_t_shape = "";
    for (auto &d : otherTy.getSizes())
      other_t_shape += std::to_string(d) + "x";
    other_t_shape.pop_back();

    uint64_t other_bytes       = xilinx::xten::getTensorVolume(otherTy);
    uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);

    std::string other_bytes_str  = std::to_string(other_bytes);
    std::string storage_str      = std::to_string(storage_i_o_bytes + other_bytes);
    
    fillPropertiesObject({"Attributes.Other.Tensor", other_t_shape},    propertiesArray);    
    fillPropertiesObject({"Attributes.Other.type",   typeStr(otherTy)},    propertiesArray);    
    fillPropertiesObject({"Attributes.Other.Bytes",  other_bytes_str},    propertiesArray);    

    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);

  } 

  void fillPropertiesCatOp(Torch::AtenCatOp &catOp, llvm::json::Array &propertiesArray) {
    std::string storage_str      = "";

    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);
  }
  
  template<class T> 
  void fillPropertiesConvOp(T &convolutionOp, llvm::json::Array &propertiesArray,
			    bool separately_return_storage = false, uint64_t *storage_n = nullptr,
			    unsigned unfused_op_id = 0) {

      std::vector<int64_t> padding;
      std::vector<int64_t> stride;
      std::vector<int64_t> dilation;      
      unpack_int_list(convolutionOp.padding(),  padding);
      unpack_int_list(convolutionOp.stride(),   stride);
      unpack_int_list(convolutionOp.dilation(), dilation);

      Value weight = convolutionOp.weight();
      Torch::BaseTensorType weightTy = weight.getType().cast<Torch::BaseTensorType>();
      Value bias = convolutionOp.bias();
      Torch::BaseTensorType biasTy;
      bool biasIsNone = bias.getType().isa<Torch::NoneType>();
      if (not biasIsNone)
	biasTy = bias.getType().cast<Torch::BaseTensorType>();
      
      uint64_t weight_o   = weightTy.getSizes()[0];
      uint64_t weight_c   = weightTy.getSizes()[1];
      uint64_t kernel_h   = weightTy.getSizes()[2];
      uint64_t kernel_w   = weightTy.getSizes()[3];

      uint64_t total_inputs    = kernel_h * kernel_w * weight_o * weight_c;
      uint64_t bias_num_inputs = biasIsNone ? 0 : xilinx::xten::getTensorVolume(biasTy);

      uint64_t weight_bytes  = total_bytes(weightTy, total_inputs);
      uint64_t bias_bytes    = biasIsNone ? 0 : total_bytes(biasTy, bias_num_inputs);

      std::string kernel_shape      = std::to_string(kernel_h) + "," + std::to_string(kernel_w);      
      std::string weight_t_shape    = std::to_string(weight_o) + "o" + std::to_string(weight_c) +
	"c" + std::to_string(kernel_h) + "h" + std::to_string(kernel_w) + "w";
      std::string shape_bytes_str   = std::to_string(weight_bytes);
      std::string bias_t_shape      = std::to_string(bias_num_inputs);
      std::string bias_bytes_str    = std::to_string(bias_bytes);
      std::string bias_type_str     = biasIsNone ? "None" : typeStr(biasTy);
      
      fillPropertiesObject({"Attributes.Weights.Tensor", weight_t_shape},    propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.Weights.type",   typeStr(weightTy)}, propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.Weights.Bytes",  shape_bytes_str},   propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.Bias.Tensor",    bias_t_shape},      propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.Bias.type",      bias_type_str},   propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.Bias.Bytes",     bias_bytes_str},    propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);

      std::string padding_str  = vector_to_str(padding);      
      std::string stride_str   = vector_to_str(stride);
      std::string dilation_str = vector_to_str(dilation);

      fillPropertiesObject({"Attributes.kernel shape",  kernel_shape},   propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);      
      fillPropertiesObject({"Attributes.padding",       padding_str},    propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.stride",        stride_str},     propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);
      fillPropertiesObject({"Attributes.dilation",      dilation_str},   propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);

      std::map<std::string, uint64_t> layerStatsMap;
      layerStatsMap = getLayerStatsMap(convolutionOp);
      
      std::string comp_mac_str = std::to_string(layerStatsMap["ops:MAC"]); 
      fillPropertiesObject({"Computations.MAC", comp_mac_str}, propertiesArray,
			   separately_return_storage, "aten.conv2d", unfused_op_id);

      if (separately_return_storage) {
	if (storage_n)
	  *storage_n = weight_bytes + bias_bytes;
      } else {
	Value input = convolutionOp.input();
	Value output = ((Operation *) convolutionOp)->getResult(0);      
	uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
	
	std::string storage_str = std::to_string(storage_i_o_bytes + bias_bytes + weight_bytes);	
	fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);
      }
  }

  inline Value getKernelSize(Torch::AtenMaxPool2dOp maxPoolOp) {
    return maxPoolOp.kernel_size();
  }

  inline Value getStride(Torch::AtenMaxPool2dOp maxPoolOp) {
    return maxPoolOp.stride();
  }

  inline Value getPadding(Torch::AtenMaxPool2dOp maxPoolOp) {
    return maxPoolOp.padding();
  }

  inline Value getKernelSize(xten::Conv2dLReLUMaxPoolOp maxPoolOp) {
    return maxPoolOp.mp_kernel_size();
  }

  inline Value getStride(xten::Conv2dLReLUMaxPoolOp maxPoolOp) {
    return maxPoolOp.mp_stride();
  }

  inline Value getPadding(xten::Conv2dLReLUMaxPoolOp maxPoolOp) {
    return maxPoolOp.mp_padding();
  }

  template<class T>
  void fillPropertiesMaxPool2dOp(T &maxPool2dOp, llvm::json::Array &propertiesArray,
				 bool separately_return_storage = false, uint64_t *storage_n = nullptr,
				 unsigned unfused_op_id = 0) {
      Value input;
      Value output;
      Value kernel_shape_v;
      Value padding_v;
      Value stride_v;

      auto mOp       = maxPool2dOp;
      input          = getInput(mOp);
      kernel_shape_v = getKernelSize(mOp);
      stride_v       = getStride(mOp);
      padding_v      = getPadding(mOp);

      output = ((Operation *)maxPool2dOp)->getResult(0);

      uint64_t maxpool_storage_bytes = 0;
      std::vector<int64_t> kernel_shape;
      std::vector<int64_t> padding;
      std::vector<int64_t> stride;

      unpack_int_list(kernel_shape_v,  kernel_shape);
      unpack_int_list(padding_v,  padding);
      unpack_int_list(stride_v,  stride);

      ////////////////For DEMO
      if (padding[0] != padding[1] and padding[0] == 0) {
      	padding.push_back(0);
      	padding.push_back(1);
      }
      ///////////////////////
      
      std::string kernel_str   = vector_to_str(kernel_shape);
      std::string padding_str  = vector_to_str(padding);      
      std::string stride_str   = vector_to_str(stride);
      
      fillPropertiesObject({"Attributes.kernel shape", kernel_str},    propertiesArray, separately_return_storage, "aten.max_pool2d");      
      fillPropertiesObject({"Attributes.padding",      padding_str},   propertiesArray, separately_return_storage, "aten.max_pool2d");
      fillPropertiesObject({"Attributes.stride",       stride_str},    propertiesArray, separately_return_storage, "aten.max_pool2d");

      std::map<std::string, uint64_t> layerStatsMap;
      layerStatsMap = getLayerStatsMap(maxPool2dOp);           
      
      fillPropertiesObject({"Computations.Vec MAX", std::to_string(layerStatsMap["ops:>"])}, propertiesArray, separately_return_storage, "max_pool2d");

      if (separately_return_storage) {
	if (storage_n)
	  *storage_n = maxpool_storage_bytes;
      } else {
	Value input  = getInput(maxPool2dOp);
	Value output = ((Operation *) maxPool2dOp)->getResult(0); 	
	uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);

	std::string storage_str  = std::to_string(maxpool_storage_bytes + storage_i_o_bytes);	
	fillPropertiesObject({"Storage.Bytes",    storage_str},       propertiesArray);
      }      
  }

  inline Value getAlpha(Torch::AtenReluOp reluOp) {
    return reluOp.self();
  }

  inline Value getAlpha(xten::Conv2dLReLUMaxPoolOp reluOp) {
    return reluOp.alpha();
  }

  inline Value getAlpha(xten::Conv2dLReLUOp reluOp) {
    return reluOp.alpha();
  }

  inline Value getInput(Torch::AtenConv2dOp cOp) {
    return cOp.input();
  }
  
  inline Value getInput(Torch::AtenMaxPool2dOp cOp) {
    return cOp.self();
  }
  
  inline Value getInput(Torch::AtenReluOp cOp) {
    return cOp.self();
  }

  inline Value getInput(Torch::AtenAddTensorOp cOp) {
    return cOp.self();
  }
  
  inline Value getInput(xten::Conv2dLReLUMaxPoolOp cOp) {
    return cOp.input();
  }

  inline Value getInput(xten::Conv2dOp cOp) {
    return cOp.input();
  }

  inline Value getInput(xten::Conv2dReLUOp cOp) {
    return cOp.input();
  }
  
  inline Value getInput(xten::Conv2dLReLUOp cOp) {
    return cOp.input();
  }
  
  template<class T>    
  void fillPropertiesReLUOp(T &reluOp, llvm::json::Array &propertiesArray,
			    bool separately_return_storage = false, uint64_t *storage_n = nullptr,
			    unsigned unfused_op_id = 0) {
    uint64_t relu_storage_bytes = 0;
    bool isLeakyReLU = not isa<Torch::AtenReluOp>(reluOp);    
    if (isLeakyReLU) {
	std::string alpha_str;
	std::string alpha_type_str;

      	Value alpha = getAlpha(reluOp); 
	uint64_t alpha_bytes = 0;
	  
	if (auto co = alpha.getDefiningOp<Torch::ConstantFloatOp>()) {
	  alpha_bytes  = sizeof(double);
	  auto alpha_n = co.value().convertToDouble();

	  alpha_str = std::to_string(alpha_n);
	  alpha_type_str = "float" + std::to_string(alpha_bytes);
	} else {
	  std::vector<int64_t> alpha_vec;	  
	  unpack_int_list(alpha, alpha_vec);
	  alpha_bytes = alpha.getType().dyn_cast<const IntegerType>().getWidth();

	  alpha_str = alpha_vec.size() != 0 ? std::to_string(alpha_vec[0]) : "";	  
	  alpha_type_str = "int" + std::to_string(alpha_bytes);
	}
	
	alpha_bytes /= BYTE_SIZE_IN_BIT;
	
	std::string alpha_t_shape   = std::to_string(1);
	std::string alpha_bytes_str = std::to_string(alpha_bytes);

	fillPropertiesObject({"Attributes.Alpha.Tensor", alpha_t_shape},   propertiesArray,
			     separately_return_storage, "aten.lrelu", unfused_op_id);
	fillPropertiesObject({"Attributes.Alpha.type",   alpha_type_str},  propertiesArray,
			     separately_return_storage, "aten.lrelu", unfused_op_id);
	fillPropertiesObject({"Attributes.Alpha.Bytes",  alpha_bytes_str}, propertiesArray,
			     separately_return_storage, "aten.lrelu", unfused_op_id);
			     fillPropertiesObject({"Attributes.Alpha",    alpha_str},  propertiesArray,
			     separately_return_storage, "aten.lrelu", unfused_op_id); 

	relu_storage_bytes += alpha_bytes;
      }  
    
      std::map<std::string, uint64_t> layerStatsMap;
      layerStatsMap = getLayerStatsMap(reluOp);

      std::string comp_mac_str = std::to_string(layerStatsMap["ops:>"]);
      fillPropertiesObject({"Computations.Comparison", comp_mac_str}, propertiesArray,
			   separately_return_storage, isLeakyReLU ? "aten.lrelu" : "aten.relu", unfused_op_id);

      if (separately_return_storage) {
	if (storage_n)
	  *storage_n = relu_storage_bytes;
      } else {
	Value input  = getInput(reluOp);
	Value output = ((Operation *) reluOp)->getResult(0); 	
	uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);

	std::string storage_str  = std::to_string(relu_storage_bytes + storage_i_o_bytes);	
	fillPropertiesObject({"Storage.Bytes",    storage_str},       propertiesArray);
      }      
  }

  void fillPropertiesLinearOp(Torch::AtenLinearOp &linearOp, llvm::json::Array &propertiesArray) {
      Value weight = linearOp.weight(); 
      Value bias   = linearOp.bias(); 
 
      Torch::BaseTensorType weightTy = weight.getType().cast<Torch::BaseTensorType>();      
      Torch::BaseTensorType biasTy   = bias.getType().cast<Torch::BaseTensorType>();
      
      uint64_t weight_num_inputs = xilinx::xten::getTensorVolume(weightTy); 
      uint64_t bias_num_inputs   = xilinx::xten::getTensorVolume(biasTy); 
      
      uint64_t weight_bytes  = total_bytes(weightTy, weight_num_inputs);
      uint64_t bias_bytes    = total_bytes(biasTy, bias_num_inputs);
      
      std::string weight_t_shape    = std::to_string(weight_num_inputs);
      std::string weight_bytes_str  = std::to_string(weight_bytes);
      std::string bias_t_shape      = std::to_string(bias_num_inputs);
      std::string bias_bytes_str    = std::to_string(bias_bytes);
      
      fillPropertiesObject({"Attributes.Weights.Tensor", weight_t_shape},    propertiesArray);
      fillPropertiesObject({"Attributes.Weights.type",   typeStr(weightTy)}, propertiesArray);
      fillPropertiesObject({"Attributes.Weights.Bytes",  weight_bytes_str},  propertiesArray);
      fillPropertiesObject({"Attributes.Bias.Tensor",    bias_t_shape},      propertiesArray);
      fillPropertiesObject({"Attributes.Bias.type",      typeStr(biasTy)},   propertiesArray);
      fillPropertiesObject({"Attributes.Bias.Bytes",     bias_bytes_str},    propertiesArray);

      Value input  = linearOp.input();
      Value output = linearOp.getResult();

      uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
      std::string storage_str = std::to_string(storage_i_o_bytes);
      
      fillPropertiesObject({"Storage.Bytes",    storage_str},   propertiesArray);            
  }

  inline Value getBnWeight(Torch::AtenBatchNormOp bnOp) {
    return bnOp.weight();
  }

  inline Value getBnWeight(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.bn_weight();
  }
  
  inline Value getBnBias(Torch::AtenBatchNormOp bnOp) {
    return bnOp.bias();
  }

  inline Value getBnBias(xten::Conv2dBatchNormReLUOp xtenBnOp) {
    return xtenBnOp.bn_bias();
  }

  template<class T>
  void fillPropertiesBatchNormOp(T &batchNormOp, llvm::json::Array &propertiesArray,
				 bool separately_return_storage = false, uint64_t *storage_n = nullptr,
				 unsigned unfused_op_id = 0) {
      Value weight = getBnWeight(batchNormOp);
      Value bias   = getBnBias(batchNormOp);
 
      Torch::BaseTensorType weightTy = weight.getType().cast<Torch::BaseTensorType>();      
      Torch::BaseTensorType biasTy   = bias.getType().cast<Torch::BaseTensorType>();

      Value r_mean      = batchNormOp.running_mean();
      Torch::BaseTensorType meanTy = r_mean.getType().cast<Torch::BaseTensorType>();
      Value r_var       = batchNormOp.running_var();
      Torch::BaseTensorType varTy  = r_var.getType().cast<Torch::BaseTensorType>();

      uint64_t weight_num_inputs = xilinx::xten::getTensorVolume(weightTy); 
      uint64_t bias_num_inputs   = xilinx::xten::getTensorVolume(biasTy); 
      uint64_t mean_num_inputs   = xilinx::xten::getTensorVolume(meanTy); 
      uint64_t var_num_inputs    = xilinx::xten::getTensorVolume(varTy); 
      
      uint64_t weight_bytes  = total_bytes(weightTy, weight_num_inputs);
      uint64_t bias_bytes    = total_bytes(biasTy, bias_num_inputs);
      uint64_t mean_bytes    = total_bytes(meanTy, mean_num_inputs);
      uint64_t var_bytes     = total_bytes(varTy, var_num_inputs);
      
      std::string weight_t_shape    = std::to_string(weight_num_inputs);
      std::string weight_bytes_str  = std::to_string(weight_bytes);
      std::string bias_t_shape      = std::to_string(bias_num_inputs);
      std::string bias_bytes_str    = std::to_string(bias_bytes);
      std::string mean_t_shape      = std::to_string(mean_num_inputs);
      std::string mean_bytes_str    = std::to_string(mean_bytes);
      std::string var_t_shape       = std::to_string(var_num_inputs);
      std::string var_bytes_str     = std::to_string(var_bytes);
      
      fillPropertiesObject({"Attributes.Weights.Tensor", weight_t_shape},    propertiesArray);
      fillPropertiesObject({"Attributes.Weights.type",   typeStr(weightTy)}, propertiesArray);
      fillPropertiesObject({"Attributes.Weights.Bytes",  weight_bytes_str},  propertiesArray);
      fillPropertiesObject({"Attributes.Bias.Tensor",    bias_t_shape},      propertiesArray);
      fillPropertiesObject({"Attributes.Bias.type",      typeStr(biasTy)},   propertiesArray);
      fillPropertiesObject({"Attributes.Bias.Bytes",     bias_bytes_str},    propertiesArray);

      fillPropertiesObject({"Attributes.Mean.Tensor",    mean_t_shape},      propertiesArray);
      fillPropertiesObject({"Attributes.Mean.type",      typeStr(meanTy)},   propertiesArray);
      fillPropertiesObject({"Attributes.Mean.Bytes",     mean_bytes_str},    propertiesArray);
      fillPropertiesObject({"Attributes.Variance.Tensor",    var_t_shape},      propertiesArray);
      fillPropertiesObject({"Attributes.Variance.type",      typeStr(varTy)},   propertiesArray);
      fillPropertiesObject({"Attributes.Variance.Bytes",     var_bytes_str},    propertiesArray);

      std::string momentum_str;
      std::string eps_str;

      Value momentum   = batchNormOp.momentum();
      Value eps        = batchNormOp.eps();

      auto momentumOp  = momentum.getDefiningOp<Torch::ConstantFloatOp>();	  
      auto momentum_n  = momentumOp.value().convertToDouble(); 
      momentum_str     = std::to_string(momentum_n);

      auto epsOp  = eps.getDefiningOp<Torch::ConstantFloatOp>();
      auto eps_n  = epsOp.value().convertToDouble(); 
      eps_str     = std::to_string(eps_n);

      fillPropertiesObject({"Attributes.eps",       eps_str},       propertiesArray);      
      fillPropertiesObject({"Attributes.momentum",  momentum_str},  propertiesArray);            

      if (separately_return_storage) {
	if (storage_n)
	  *storage_n = mean_bytes + var_bytes + weight_bytes + bias_bytes;
      } else {
	Value input  = batchNormOp.input();      
	Value output = ((Operation *) batchNormOp)->getResult(0); 
	uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);	
	std::string storage_str  = std::to_string(var_bytes + mean_bytes + storage_i_o_bytes +
						  weight_bytes + bias_bytes);	
	fillPropertiesObject({"Storage.Bytes",    storage_str},       propertiesArray);
      }
  } 
  
  template<class T>
  void fillPropertiesSoftmaxOp(T &softmaxOp, llvm::json::Array &propertiesArray) {
      Value dim           = softmaxOp.dim();
      uint64_t dim_v      = dim.getDefiningOp<mlir::arith::ConstantIntOp>().value();      
      std::string dim_str = std::to_string(dim_v);

      Value input  = softmaxOp.self();
      Value output = softmaxOp.getResult();

      uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
      std::string storage_str = std::to_string(storage_i_o_bytes);
      
      fillPropertiesObject({"Attributes.dim",    dim_str},      propertiesArray);
      fillPropertiesObject({"Storage.Bytes",    storage_str},   propertiesArray);      
  }

  void fillPropertiesGatherOp(Torch::AtenGatherOp &gatherOp, llvm::json::Array &propertiesArray) {
      Value dim           =  gatherOp.dim();
      uint64_t dim_v      = dim.getDefiningOp<mlir::arith::ConstantIntOp>().value();      
      std::string dim_str = std::to_string(dim_v);

      Value index = gatherOp.index();
      Torch::BaseTensorType indexTy = index.getType().cast<Torch::BaseTensorType>();      

      uint64_t index_num_inputs = xilinx::xten::getTensorVolume(indexTy);       
      uint64_t index_bytes      = total_bytes(indexTy, index_num_inputs);      

      std::string index_t_shape        = std::to_string(index_num_inputs);
      std::string index_bytes_str      = std::to_string(index_bytes);
      
      Value input  = gatherOp.self();
      Value output = gatherOp.getResult();

      uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
      std::string storage_str = std::to_string(storage_i_o_bytes);
      
      fillPropertiesObject({"Attributes.dim",    dim_str},      propertiesArray);

      fillPropertiesObject({"Attributes.Index.Tensor", index_t_shape},    propertiesArray);
      fillPropertiesObject({"Attributes.Index.type",   typeStr(indexTy)}, propertiesArray);
      fillPropertiesObject({"Attributes.Index.Bytes",  index_bytes_str},  propertiesArray);

      fillPropertiesObject({"Storage.Bytes",    storage_str},   propertiesArray);      
  }

  void fillPropertiesSliceOp(Torch::AtenSliceTensorOp &sliceOp, llvm::json::Array &propertiesArray) {
      Value dim           = sliceOp.dim();
      uint64_t dim_v      = dim.getDefiningOp<mlir::arith::ConstantIntOp>().value();      
      std::string dim_str = std::to_string(dim_v);

      Value input  = sliceOp.self();
      Value output = sliceOp.getResult();

      uint64_t storage_i_o_bytes = storage_bytes_of_input_and_output(input, output);
      std::string storage_str = std::to_string(storage_i_o_bytes);
      
      fillPropertiesObject({"Attributes.dim",    dim_str},      propertiesArray);
      fillPropertiesObject({"Storage.Bytes",    storage_str},   propertiesArray);      
  }
  
  void fillPropertiesOp(xten::Conv2dBatchNormReLUOp &xtenConv2dBnReluOp, llvm::json::Array &propertiesArray) {
    uint64_t conv_storage;
    uint64_t bn_storage;
    uint64_t relu_storage = 0;

    fillPropertiesConvOp<xten::Conv2dBatchNormReLUOp>(xtenConv2dBnReluOp, propertiesArray,
						      true, &conv_storage, 0);

    fillPropertiesBatchNormOp<xten::Conv2dBatchNormReLUOp>(xtenConv2dBnReluOp, propertiesArray,
    							   true, &bn_storage, 1);

    //fillPropertiesReLUOp<xten::Conv2dBatchNormReLUOp>(xtenConv2dBnReluOp, propertiesArray,
    //						      true, &relu_storage, 2);

    uint64_t storage = conv_storage + bn_storage + relu_storage;

    Value input  = xtenConv2dBnReluOp.input();
    Value output = xtenConv2dBnReluOp.getResult();
    storage += storage_bytes_of_input_and_output(input, output);

    std::string storage_str = std::to_string(storage);
    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);      
  } 

  void fillPropertiesOp(xten::Conv2dLReLUMaxPoolOp &xtenConv2LReluMaxPoolOp, llvm::json::Array &propertiesArray) {
    uint64_t conv_storage;
    uint64_t lrelu_storage;
    uint64_t maxpool_storage;

    fillPropertiesConvOp<xten::Conv2dLReLUMaxPoolOp>(xtenConv2LReluMaxPoolOp, propertiesArray,
						     true, &conv_storage, 0);
    ///////// For DEMO
    if (propagate_tensor_sizes) {
      auto tensor_size = propagatedTensorSizesMap[xtenConv2LReluMaxPoolOp][true]["value"];    
      std::string tensor_size_str = std::to_string(tensor_size[0]) + "n" + std::to_string(tensor_size[1]) + "c" +
	std::to_string(tensor_size[2]) + "h" + std::to_string(tensor_size[3]) + "w";
      fusedToUnfuseOutputMap[xtenConv2LReluMaxPoolOp]["aten.conv2d"] = tensor_size_str; 
    }
    ///////////////////////////////////////

    
    fillPropertiesReLUOp<xten::Conv2dLReLUMaxPoolOp>(xtenConv2LReluMaxPoolOp, propertiesArray,
						     true, &lrelu_storage, 1);
    
    fillPropertiesMaxPool2dOp<xten::Conv2dLReLUMaxPoolOp>(xtenConv2LReluMaxPoolOp, propertiesArray,
    							  true, &maxpool_storage, 2);
    ///////// For DEMO
    if (propagate_tensor_sizes) {
      auto tensor_size = propagatedTensorSizesMap[xtenConv2LReluMaxPoolOp][false]["value"];
      std::string tensor_size_str = std::to_string(tensor_size[0]) + "n" + std::to_string(tensor_size[1]) + "c" +
	std::to_string(tensor_size[2]) + "h" + std::to_string(tensor_size[3]) + "w";
      fusedToUnfuseOutputMap[xtenConv2LReluMaxPoolOp]["aten.max_pool2d"] = tensor_size_str; 
    }
    ///////////////////////////////////////
    
    uint64_t storage = conv_storage + maxpool_storage + lrelu_storage;

    Value input  = xtenConv2LReluMaxPoolOp.input(); 
    Value output = xtenConv2LReluMaxPoolOp.getResult();
    storage += storage_bytes_of_input_and_output(input, output);

    std::string storage_str = std::to_string(storage);
    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);      
  } 

  void fillPropertiesOp(xten::Conv2dLReLUOp &xtenConv2dLReluOp, llvm::json::Array &propertiesArray) {
    uint64_t conv_storage;
    uint64_t lrelu_storage;
    
    fillPropertiesConvOp<xten::Conv2dLReLUOp>(xtenConv2dLReluOp, propertiesArray,
					      true, &conv_storage, 0);

    /////// For DEMO
    if (propagate_tensor_sizes) {
      auto tensor_size = propagatedTensorSizesMap[xtenConv2dLReluOp][false]["value"];    
      std::string tensor_size_str = std::to_string(tensor_size[0]) + "n" + std::to_string(tensor_size[1]) + "c" +
	std::to_string(tensor_size[2]) + "h" + std::to_string(tensor_size[3]) + "w";
      fusedToUnfuseOutputMap[xtenConv2dLReluOp]["aten.conv2d"] = tensor_size_str; 
    }
    ///////////////////////////////////////
    
    fillPropertiesReLUOp<xten::Conv2dLReLUOp>(xtenConv2dLReluOp, propertiesArray,
    					      true, &lrelu_storage, 1);
    
    uint64_t storage = conv_storage + lrelu_storage;

    Value input  = xtenConv2dLReluOp.input();
    Value output = xtenConv2dLReluOp.getResult();
    storage += storage_bytes_of_input_and_output(input, output);

    std::string storage_str = std::to_string(storage);
    fillPropertiesObject({"Storage.Bytes",    storage_str},  propertiesArray);      
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
      operatorObject["name"]        = op_str;
      operatorObject["description"] = "ATen/XTen operator " + op_str;
      
      llvm::json::Array operatorPropsArray;
      for (auto const &op_props_vec : propertiesInfo[op_str]) {
	llvm::json::Object operatorPropsObject;
	operatorPropsObject["name"]          =  op_props_vec[0];
	operatorPropsObject["tooltip"]       =  op_props_vec[1];
	operatorPropsObject["type"]          =  op_props_vec[2];

	operatorPropsArray.push_back(llvm::json::Value(std::move(operatorPropsObject)));
      }
	  
      operatorObject["properties"]  = llvm::json::Value(std::move(operatorPropsArray));	
      operatorsArray.push_back(llvm::json::Value(std::move(operatorObject)) );
    }

    return operatorsArray;
  }
  
  llvm::json::Array emitJSONDesigns() {
    llvm::json::Array designsArray;

    for (auto const &design_pair : designToLayers) {
      llvm::json::Object designObject;
      llvm::json::Object propertyObject;
      llvm::json::Array  propertyArray;
      
      int design_id   = design_pair.first;
      int layer_count = design_pair.second.size();

      propertyObject["name"]    = "Layer Count";
      propertyObject["tooltip"] = "Total number of layers in design";
      propertyObject["type"]    = "int";
      propertyObject["value"]   = std::to_string(layer_count);
      propertyArray.push_back(llvm::json::Value(std::move(propertyObject)));

      designObject["name"] = "design " + std::to_string(design_id);
      designObject["properties"] = llvm::json::Value(std::move(propertyArray));
      
      designsArray.push_back(llvm::json::Value(std::move(designObject)));
    }

    return designsArray;
  }

  llvm::json::Array fillProperties(Operation *op) {
    llvm::json::Array propertiesArray;
    if (auto convolutionOp      = dyn_cast<Torch::AtenConv2dOp>(op)) {
      fillPropertiesConvOp<Torch::AtenConv2dOp>(convolutionOp, propertiesArray);
    } else if (auto maxPoolOp   = dyn_cast<Torch::AtenMaxPool2dOp>(op)) {
      fillPropertiesMaxPool2dOp<Torch::AtenMaxPool2dOp>(maxPoolOp, propertiesArray);
    } else if (auto reluOp      = dyn_cast<Torch::AtenReluOp>(op)) {
      fillPropertiesReLUOp<Torch::AtenReluOp>(reluOp, propertiesArray);
    } else if (auto batchNormOp = dyn_cast<Torch::AtenBatchNormOp>(op)) {
      fillPropertiesBatchNormOp<Torch::AtenBatchNormOp>(batchNormOp, propertiesArray);
    } else if (auto linearOp    = dyn_cast<Torch::AtenLinearOp>(op)) {
      fillPropertiesLinearOp(linearOp, propertiesArray);
    } else if (auto sizeOp      = dyn_cast<Torch::AtenSizeOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSizeOp>(sizeOp, propertiesArray);
    } else if (auto catOp      = dyn_cast<Torch::AtenCatOp>(op)) {
      fillPropertiesCatOp(catOp, propertiesArray);
    } else if (auto negOp       = dyn_cast<Torch::AtenNegOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenNegOp>(negOp, propertiesArray);
    } else if (auto sigmoidOp   = dyn_cast<Torch::AtenSigmoidOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSigmoidOp>(sigmoidOp, propertiesArray);
    } else if (auto sinOp       = dyn_cast<Torch::AtenSinOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenSinOp>(sinOp, propertiesArray);
    } else if (auto tanhOp      = dyn_cast<Torch::AtenTanhOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenTanhOp>(tanhOp, propertiesArray);
    } else if (auto expOp       = dyn_cast<Torch::AtenExpOp>(op)) {
      fillPropertiesUnaryALUOp<Torch::AtenExpOp>(expOp, propertiesArray);
    } else if (auto addOp       = dyn_cast<Torch::AtenAddTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenAddTensorOp>(addOp, propertiesArray);
    } else if (auto mulOp       = dyn_cast<Torch::AtenMulTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenMulTensorOp>(mulOp, propertiesArray);
    } else if (auto divOp       = dyn_cast<Torch::AtenDivTensorOp>(op)) {
      fillPropertiesBinaryALUOp<Torch::AtenDivTensorOp>(divOp, propertiesArray);
    } else if (auto gatherOp    = dyn_cast<Torch::AtenGatherOp>(op)) {
      fillPropertiesGatherOp(gatherOp, propertiesArray);
    } else if (auto sliceOp     = dyn_cast<Torch::AtenSliceTensorOp>(op)) {
      fillPropertiesSliceOp(sliceOp, propertiesArray);
    } else if (auto xtenConv2dOp             = mlir::dyn_cast<xten::Conv2dOp>(op)) {
      fillPropertiesConvOp<xten::Conv2dOp>(xtenConv2dOp, propertiesArray);          
    } else if (auto xtenConv2dBnReluOp       = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
      fillPropertiesOp(xtenConv2dBnReluOp, propertiesArray);      
    } else if (auto xtenConv2dLReluOp        = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
      fillPropertiesOp(xtenConv2dLReluOp, propertiesArray);      
    } else if (auto xtenConv2dLReluMaxPoolOp = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      fillPropertiesOp(xtenConv2dLReluMaxPoolOp, propertiesArray);      
    } else if (auto xtenAddOp                = mlir::dyn_cast<xten::AddOp>(op)  ) {
      //fillPropertiesBinaryALUOp<xten::AddOp>(xtenAddOp, propertiesArray); 
    }

    return propertiesArray;
  }

  Value getInput(Operation *op) {
    Value opInput;

    if (auto convolutionOp      = dyn_cast<Torch::AtenConv2dOp>(op)) {
      opInput = getInput(convolutionOp);
    } else if (auto maxPoolOp   = dyn_cast<Torch::AtenMaxPool2dOp>(op)) {
      opInput = getInput(maxPoolOp);
    } else if (auto reluOp      = dyn_cast<Torch::AtenReluOp>(op)) {
      opInput = getInput(reluOp);
    } else if (auto addOp       = dyn_cast<Torch::AtenAddTensorOp>(op)) {
      opInput = getInput(addOp); //TODO: add Tensor technically has two inputs 
    } else if (auto xtenConv2dOp              = mlir::dyn_cast<xten::Conv2dOp>(op)) {
      opInput = getInput(xtenConv2dOp);
    } else if (auto xtenConv2dBnReluOp        = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
      opInput = getInput(xtenConv2dBnReluOp);
    } else if (auto xtenConv2dLReluOp         = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
      opInput = getInput(xtenConv2dLReluOp);
    } else if (auto xtenConv2dLReluMaxPoolOp  = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      opInput = getInput(xtenConv2dLReluMaxPoolOp);
    }
    //TODO: expand switch table for more ops
    
    return opInput;
  }

  //////////// For DEMO
  std::vector<int64_t> getOutputTensorDivFactors(Operation *op) {
    std::vector<int64_t> n_stride = {1, 1};
    
    if (auto xtenConv2dOp = mlir::dyn_cast<xten::Conv2dOp>(op)) {
    } else if (auto xtenConv2dBnReluOp = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
    } else if (auto xtenConv2dLReluOp = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
    } else if (auto xtenConv2dLReluMaxPoolOp = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      Value stride  = xtenConv2dLReluMaxPoolOp.mp_stride();
      std::vector<int64_t> v_stride;
      unpack_int_list(stride,  v_stride);  
      n_stride = v_stride;
    } 
    return n_stride;
  }

  uint64_t getChannelsOut(Operation *op) {
    uint64_t cout = 0;
    Value weight;
    if (auto xtenConv2dOp = mlir::dyn_cast<xten::Conv2dOp>(op)) {
      weight  = xtenConv2dOp.weight();
    } else if (auto xtenConv2dBnReluOp = mlir::dyn_cast<xten::Conv2dBatchNormReLUOp>(op)) {
      weight  = xtenConv2dBnReluOp.weight();
    } else if (auto xtenConv2dLReluOp = mlir::dyn_cast<xten::Conv2dLReLUOp>(op)) {
      weight  = xtenConv2dLReluOp.weight();
    } else if (auto xtenConv2dLReluMaxPoolOp = mlir::dyn_cast<xten::Conv2dLReLUMaxPoolOp>(op)) {
      weight  = xtenConv2dLReluMaxPoolOp.weight();
    } else {
      return 0;
    }

    Torch::BaseTensorType weightTy = weight.getType().cast<Torch::BaseTensorType>();    
    cout   = weightTy.getSizes()[0];    
    return cout;
  }
  ///////////////////////////////////////////////
  
  void fillPortProperties(Operation *op, bool isInput, llvm::json::Array &portPropsArray,
			  bool unfused_part_of_fused = false, std::string unfused_part_name = "",
			  unsigned unfused_id = 0) {
    std::string port_name_prefix;
    std::string port_type_str;
    if (isInput) {
      port_name_prefix = "Inputs.IFMs";
      port_type_str    = "Input";
    }
    else {
      port_name_prefix = "Outputs.OFMs";
      port_type_str    = "Output";
    }
  
    llvm::json::Object portPropsObject;    
    if (unfused_part_of_fused) {
      portPropsObject["name"]    = port_name_prefix + ".Tensor";
      portPropsObject["tooltip"] = "Dimensions of " + port_type_str;
      portPropsObject["type"]    = "string";    

      /* fusedToUnfuseOutputMaps should have been filled by previous fillProperties functions */
      portPropsObject["value"] = fusedToUnfuseOutputMap[op][unfused_part_name];

      portPropsObject["unfused_operator_type"]  = unfused_part_name;
      portPropsObject["unfused_operator_id"]    = std::to_string(unfused_id);    
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));
      
    } else {    
      std::string value_str = "";
      std::string type_str  = "";
      std::string bytes_str = "";
      uint64_t total_inputs = 0;

      //TODO: for operators with multiple outputs, will need to go over them
      Type resultTy = (inputOpToIFMValue.find(op) != inputOpToIFMValue.end() and isInput ? inputOpToIFMValue[op] : op->getResult(0)).getType();
      Torch::BaseTensorType sizeResultTy = resultTy.dyn_cast<Torch::BaseTensorType>();
      if (not sizeResultTy)
	return;
      
      if (sizeResultTy.hasSizes()) {
	std::vector<int64_t> size_vec;
	total_inputs = 1;
	for (auto dim : sizeResultTy.getSizes()) {
	  size_vec.push_back(dim);
	  total_inputs *= dim;
	}
	
	value_str = vector_to_str(size_vec, "x");
	type_str  = typeStr(sizeResultTy);
	bytes_str = std::to_string(total_bytes(sizeResultTy, total_inputs));	
      }
      
      ///////// For TinyYolo-DEMO only
      if (propagate_tensor_sizes) {
	auto value_v   = propagatedTensorSizesMap[op][isInput]["value"];
	auto bit_width = propagatedTensorSizesMap[op][isInput]["type"][0];
	uint64_t n, c, h, w;
	n = value_v[0];
	c = value_v[1];
	h = value_v[2];
	w = value_v[3];
	total_inputs = n * c * h * w;
	
	value_str = std::to_string(n) + "n" + std::to_string(c) +
	  "c" + std::to_string(h) + "h" + std::to_string(w) + "w";
	type_str  = propagatedTensorTypeMap[op];
	bytes_str = std::to_string( (bit_width/BYTE_SIZE_IN_BIT) * total_inputs); 
      }
      /////////////////////////////////////////
      
      portPropsObject["name"]    = port_name_prefix + ".Tensor";
      portPropsObject["tooltip"] = "Dimensions of " + port_type_str;
      portPropsObject["type"]    = "string";    
      portPropsObject["value"]   = value_str;            
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));
    
      portPropsObject["name"]    = port_name_prefix + ".type";
      portPropsObject["tooltip"] = "Numerical type of each element of " + port_type_str;
      portPropsObject["type"]    = "string";
      portPropsObject["value"]   = type_str;       
      portPropsArray.push_back(llvm::json::Value(std::move(portPropsObject)));
    
      portPropsObject["name"]    = port_name_prefix + ".Bytes";
      portPropsObject["tooltip"] = "Size of " + port_type_str + " in bytes";
      portPropsObject["type"]    = "long";
      portPropsObject["value"]   = bytes_str;       
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

      ///////// For DEMO only
      if (propagate_tensor_sizes) {
	fillPortProperties(op, true, portPropsArray);

	/*For each fused operator, also add to JSON model sub-outputs of unfused parts of fused op  */
	auto op_name = getOperationNameStr(op);
	if (fusedOpToUnfusedOpsMap.find(op_name) != fusedOpToUnfusedOpsMap.end()) {
	  unsigned unfused_id = 1;
	  for (auto unfused_op_name : fusedOpToUnfusedOpsMap[op_name]) {
	    fillPortProperties(op, true, portPropsArray, true, unfused_op_name, unfused_id++);
	  }
	}
      } ////////////////////////////////////////////////
      else 
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

      /*For each fused operator, also add to JSON model sub-outputs of unfused parts of fused op  */
      auto op_name = getOperationNameStr(op);
      if (fusedOpToUnfusedOpsMap.find(op_name) != fusedOpToUnfusedOpsMap.end()) {
	unsigned unfused_id = 1;
	for (auto unfused_op_name : fusedOpToUnfusedOpsMap[op_name]) {
	  fillPortProperties(op, false, portPropsArray, true, unfused_op_name, unfused_id++);
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
      llvm::json::Array  propertiesArray;
      llvm::json::Array  operatorsArray;

      auto layer_name = layer_pair.first;
      auto layer_ops  = layer_pair.second;
      int layer_ops_count = layer_ops.size();

      propertyObject["name"]    = "Operator Count";
      propertyObject["tooltip"] = "Total number of ATen operators in layer to be visualized";
      propertyObject["type"]    = "int";
      propertyObject["value"]   = std::to_string(layer_ops_count);
      
      layerObject["name"]        = layer_name;
      layerObject["design_name"] = "design " + std::to_string(currentDesign); 

      propertiesArray.push_back(llvm::json::Value(std::move(propertyObject)));
      layerObject["properties"] = llvm::json::Value(std::move(propertiesArray));
      for (auto op : layer_ops) {
	llvm::json::Object operatorObject;
	auto op_id_str = opToName[op].first;

	operatorObject["id"]             = std::to_string(opToId[op]);
	operatorObject["name"]           = op_id_str;
	operatorObject["operator_type"]  = getOperationNameStr(op);

	auto propertiesArray = fillProperties(op);
	operatorObject["properties"]  = llvm::json::Value(std::move(propertiesArray));

	auto portsArray = emitJSONLayerOpPorts(op);
	operatorObject["ports"]       = llvm::json::Value(std::move(portsArray));

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
      Operation *input_op    = conns_pair.first;
      const auto output_ops  = conns_pair.second;
      for (auto const &output_op_pair : output_ops) {
	Operation *output_op = output_op_pair.first;
	if (output_op == input_op)
	  continue;

	llvm::json::Object connectionObject;
      
	unsigned inPortId   = output_op_pair.second;
	unsigned outPortId  = connsOutToInMap[output_op][input_op];

	connectionObject["id"]           = std::to_string(connection_id++);
	connectionObject["from_port_id"] = std::to_string(outPortId);
	connectionObject["to_port_id"]   = std::to_string(inPortId); 
	connectionsArray.push_back(llvm::json::Value(std::move(connectionObject)));
      }
    } 

    return connectionsArray;
  }

  std::vector<Operation *> vectorValidArgOps(Operation *argOp, Operation *op) {
    auto vectorArgOps = std::vector<Operation *>();
    if (opIsValid(argOp)) 	      
      vectorArgOps.push_back(argOp);
    else if (getOperationNameStr(op)    == "torch.aten.cat" and 
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
      *this, "operators-supported-path", llvm::cl::desc("Path of JSON file that has list of operators supported (REQUIRED)"),
      llvm::cl::init("-")};

  Option<bool> ATenPropagateTensorSizesFlag{
      *this, "propagate-tensor-sizes", llvm::cl::desc("Boolean flag to indicate if pass should propagate tensor sizes (only works with Tiny Yolo)"),
      llvm::cl::init(false)};

  ATenVisualGraphPass(const ATenVisualGraphPass &pass) : output(o) {}

  ATenVisualGraphPass()
      : output(o) {}

  std::string emitJSONReport() {
    llvm::json::Object top;

    llvm::json::Object flexmlTop;
    llvm::json::Object flexmlSchema;
    llvm::json::Array  flexmlDesigns;
    llvm::json::Array  flexmlOperators;
    llvm::json::Array  flexmlLayers;
    llvm::json::Array  flexmlConnections;

    flexmlSchema      = emitJSONSchema();
    flexmlOperators   = emitJSONOperators();
    flexmlDesigns     = emitJSONDesigns();
    flexmlLayers      = emitJSONLayers();
    flexmlConnections = emitJSONConnections();

    //Fill Top JSON properties
    flexmlTop["schema_version"]   = llvm::json::Value(std::move(flexmlSchema));
    flexmlTop["designs"]          = llvm::json::Value(std::move(flexmlDesigns));
    flexmlTop["operator_types"]   = llvm::json::Value(std::move(flexmlOperators));
    flexmlTop["layers"]           = llvm::json::Value(std::move(flexmlLayers));
    flexmlTop["connections"]      = llvm::json::Value(std::move(flexmlConnections));
    
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
    propagate_tensor_sizes = ATenPropagateTensorSizesFlag;

    // I don't change anything  
    markAllAnalysesPreserved();

    auto module = getOperation();

    // check that a function called "forward" exists
    auto forward = module.lookupSymbol<mlir::FuncOp>("forward");
    if (!forward) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a forward function\n");
      signalPassFailure();
      return;
    }

    clearAllDataStructures();
    //////For DEMO
    Operation *prevOp = nullptr; 
    /////////

    unsigned currentOp  = 0;    
    unsigned currPortId = 0;    
    forward.walk([&](Operation *op) {
 	if (not opIsValid(op))
	  return;

	auto attr_l = op->getAttrOfType<StringAttr>("layer_name");
	auto attr_n = op->getAttrOfType<StringAttr>("name");
	//assumes layer_name is given to all nodes, might support infering layers later
	if (!attr_l and !attr_n)
	  return;
	auto attr = attr_l ? attr_l : attr_n;
	
	auto op_str     = getOperationNameStr(op);
	auto layer_name = attr.getValue().str();
	layerToOps[layer_name].push_back(op);
	designToLayers[currentDesign].push_back(layer_name);

	opToName[op] = std::make_pair(op_str + "_" + std::to_string(opTypes[op_str]), layer_name);
	opToId[op]   = currentOp;

	for (auto argOp_ref : op->getOperands()) {
	  if (auto argOp = argOp_ref.getDefiningOp()) {
	    auto vectorArgOps = vectorValidArgOps(argOp, op);
	    for (auto argOp : vectorArgOps) {
	      // where Op accepts ArgOp as input  <===> Op(%ArgOp, ...) 
	      connsInToOutMap[op][argOp] = currPortId++;
	      
	      // where output of ArgOp is sent to input port of Op <===>  ArgOp.output ----> Op.input_port_example
	      //same output port goes to several next input ports -> this output port should have one Port ID
	      if (connsOutToInMap[argOp].size() == 0)
		connsOutToInMap[argOp][op] = currPortId++;
	      else
		connsOutToInMap[argOp][op] = connsOutToInMap[argOp].begin()->second;
	    }
	  }
	}
       	
	// ------------- For DEMO 
	if (propagate_tensor_sizes) {
	  uint64_t o, c, h, w;
	  uint64_t bit_width;
	  std::string type_str;
	  bool isInput  = true;
	  bool isOutput = not isInput;
	  if (currentOp == 0) {
	    Value cInput = getInput(op);
	    Torch::BaseTensorType inputTy = cInput.getType().cast<Torch::BaseTensorType>();	  
	    o   = inputTy.getSizes()[0];
	    c   = inputTy.getSizes()[1];	  
	    h   = inputTy.getSizes()[2];
	    w   = inputTy.getSizes()[3];
	    bit_width = total_bytes(inputTy, 1) * BYTE_SIZE_IN_BIT;
	    type_str = typeStr(inputTy);

	  } else {
	    auto input_v = propagatedTensorSizesMap[prevOp][isOutput]["value"];
	    o = input_v[0];
	    c = input_v[1];
	    h = input_v[2];
	    w = input_v[3];
	    bit_width    = propagatedTensorSizesMap[prevOp][isOutput]["type"][0];
	    type_str     = propagatedTensorTypeMap[prevOp];
	  }
	
	  auto stride_v = getOutputTensorDivFactors(op);
	  uint64_t o2, c2, h2, w2;
	
	  o2 = o;
	  c2 = getChannelsOut(op);
	  h2 = h/stride_v[0];
	  w2 = w/stride_v[1];

	  propagatedTensorSizesMap[op][isInput]["value"]  = {o,c,h,w};	
	  propagatedTensorSizesMap[op][isOutput]["value"] = {o2,c2,h2,w2};

	  propagatedTensorSizesMap[op][isInput]["type"]  = {bit_width};	
	  propagatedTensorSizesMap[op][isOutput]["type"] = {bit_width};

	  propagatedTensorTypeMap[op] = type_str;
	
	  std::string inputStr  = std::to_string(o) + "o" + std::to_string(c) +
	    "c" + std::to_string(h) + "h" + std::to_string(w) + "w";
	  std::string outputStr = std::to_string(o2) + "o" + std::to_string(c2) +
	    "c" + std::to_string(h2) + "h" + std::to_string(w2) + "w";
 
	  prevOp = op;
	}
	//////////////////////////////////////
	
	updateOperatorTypes(op);
	currentOp++;      
    });
    
    /* for operators whose outputs are the return values of 'graph' */
    for (const auto &op_pair : opToName) {
      auto op = op_pair.first;
      if (connsInToOutMap.find(op) == connsInToOutMap.end()) {       
        //For the first input Ops (aka source Ops) in the NN graph
        connsInToOutMap[op][op] = currPortId++;
	inputOpToIFMValue[op]   = getInput(op);
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

