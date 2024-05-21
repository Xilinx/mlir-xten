//===- ATenDialectOpStats.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "xten/Util/Util.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>
#include <type_traits>

#define DEBUG_TYPE "aten-op-stats"

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;
using namespace xilinx;
using namespace mlir::torch;

namespace {

template <class T>
std::map<std::string, uint64_t>
getConv2dStatisticsWithType(T o, Torch::BaseTensorType resultTy) {
  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType inputTy =
      o.getInput().getType().template cast<Torch::BaseTensorType>();
  Torch::BaseTensorType weightTy =
      o.getWeight().getType().template cast<Torch::BaseTensorType>();
  Torch::BaseTensorType biasTy;
  if (o.getBias()) {
    biasTy = o.getBias().getType().template cast<Torch::BaseTensorType>();
  }

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  uint64_t ifm_depth = inputTy.getSizes()[1];
  uint64_t kernel_height = weightTy.getSizes()[2];
  uint64_t kernel_width = weightTy.getSizes()[3];

  auto co = o.getGroups().getDefiningOp();
  auto ia = co->template getAttrOfType<IntegerAttr>("value");
  uint64_t groups = ia.getValue().getZExtValue();
  // Number of forward MACs per pixel =
  //  kernel_width * kernel_height * ifm_depth / groups
  uint64_t MACs_per_OFM = (ifm_depth / groups) * kernel_height * kernel_width;
  uint64_t total_MACs = ofm_volume * MACs_per_OFM;

  uint64_t ifm_volume = xilinx::xten::getTensorVolume(inputTy);
  uint64_t weight_volume = xilinx::xten::getTensorVolume(weightTy);
  uint64_t bias_volume;
  if (o.getBias()) {
    bias_volume = xilinx::xten::getTensorVolume(biasTy);
  } else {
    bias_volume = 0;
  }

  // Should be gated on whether there is bias at all
  toReturn["ops:+"] = ofm_volume;

  toReturn["ops:MAC"] = total_MACs;
  toReturn["operand:0:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  toReturn["reads"] = weight_volume + bias_volume + ifm_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

static bool simple_conv2d_model = false;

template <class T>
uint64_t
getConv2dOperandTransferVolumeWithType(T o, unsigned int idx, bool read,
                                       Torch::BaseTensorType resultTy) {

  if (!read)
    return 0;

  double vol = xilinx::xten::getTensorVolume(o.getOperand(idx).getType());
  if (simple_conv2d_model)
    return vol;

  Torch::BaseTensorType inputTy =
      o.getInput().getType().template cast<Torch::BaseTensorType>();
  Torch::BaseTensorType weightTy =
      o.getWeight().getType().template cast<Torch::BaseTensorType>();

  float filter_width = weightTy.getSizes()[2];
  float filter_height = weightTy.getSizes()[3];

  float batch_sw = inputTy.getSizes()[0];
  // float ifm_depth_sw = inputTy.getSizes()[1];
  float ih = inputTy.getSizes()[2];
  // float iw = inputTy.getSizes()[3];

  float ofm_depth_sw = resultTy.getSizes()[1];

  const float batch_hw = 4;
  // const float ifm_depth_hw = 32;
  const float ofm_depth_hw = 32;

  const float ifm_tile_height = 4;
  // const float ifm_tile_width = 4;
  // const float ofm_tile_height = 4;
  // const float ofm_tile_width = 4;

  float ifm_aperture = ifm_tile_height - ceilf(filter_height / 2.0f);
  float ifm_overlap = ceilf(filter_height / 2.0f);

  float bl = ceilf(batch_sw / batch_hw);
  float ol = ceilf(ofm_depth_sw / ofm_depth_hw);
  // float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float ifm_overhead = 1.0f;
  float weight_overhead = 1.0f;
  if (filter_width > 1) {
    ifm_overhead =
        ol * ifm_tile_height * ((ih - ifm_overlap) / (ih * ifm_aperture));
    weight_overhead = bl;
  } else {
    ifm_overhead = ol;
  }

  if (idx == 0) {
    LLVM_DEBUG(llvm::outs() << "ifm_overhead:" << ifm_overhead << "\n");
    return vol * ifm_overhead;
  }
  if (idx == 1) {
    LLVM_DEBUG(llvm::outs() << "weight_overhead:" << weight_overhead << "\n");
    return vol * weight_overhead;
  }
  return vol;
}

template <class T>
uint64_t getConv2dResultTransferVolumeWithType(T o, unsigned int idx,
                                               bool write,
                                               Torch::BaseTensorType resultTy) {

  Torch::BaseTensorType inputTy =
      o.getInput().getType().template cast<Torch::BaseTensorType>();

  if (simple_conv2d_model) {
    if (write)
      return xilinx::xten::getTensorVolume(resultTy);
    else
      return 0;
  }

  Torch::BaseTensorType weightTy =
      o.getWeight().getType().template cast<Torch::BaseTensorType>();
  float filter_width = weightTy.getSizes()[2];
  // float filter_height = weightTy.getSizes()[3];

  float ifm_depth_sw = inputTy.getSizes()[1];
  const float ifm_depth_hw = 32;

  float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float write_output_overhead = 1.0f;
  float read_output_cost = 1.0f;

  if (filter_width > 1) {
    write_output_overhead = il;
    read_output_cost = il;
  }

  double vol = xilinx::xten::getTensorVolume(resultTy);

  if (write) {
    LLVM_DEBUG(llvm::outs()
               << "write_output_overhead:" << write_output_overhead << "\n");
    return vol * write_output_overhead;
  } else {
    LLVM_DEBUG(llvm::outs() << "read_output_cost:" << read_output_cost << "\n");
    return vol * read_output_cost;
  }
}

// TODO can this indirection be cleaned?

// Conv2dStatistics
template <class T>
std::map<std::string, uint64_t> getConv2dStatistics(T o) {
  Torch::BaseTensorType resultType =
      o.getResult().getType().template cast<Torch::BaseTensorType>();
  return getConv2dStatisticsWithType(o, resultType);
}

// OperandTransferVolume
template <class T>
uint64_t getConv2dOperandTransferVolume(T o, unsigned int idx, bool read) {
  Torch::BaseTensorType resultType =
      o.getResult().getType().template cast<Torch::BaseTensorType>();
  return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

// ResultTransferVolume
template <class T>
uint64_t getConv2dResultTransferVolume(T o, unsigned int idx, bool write) {
  Torch::BaseTensorType resultType =
      o.getResult().getType().template cast<Torch::BaseTensorType>();
  return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}

} // namespace

#if 1

namespace xilinx {
namespace xten {

using namespace mlir::torch;

template <class OpT>
std::map<std::string, uint64_t> getStatistics(OpT op) {
  return std::map<std::string, uint64_t>();
}

// add
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenAddTensorOp op) {
  std::map<std::string, uint64_t> toReturn;
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(aType);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// add_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenAdd_TensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// // addmm
// template<>
// std::map<std::string, uint64_t> getStatistics(AddmmOp op) {

//   std::map<std::string, uint64_t> toReturn;

//   // For linear, we need the number of output neurons and the number of input
//   neurons
//   // Then the number of forward MACs is input * output
//   // And the number of adds is output if there is bias

//   Torch::BaseTensorType resultTy =
//   op.getResult().getType().cast<Torch::BaseTensorType>();
//   Torch::BaseTensorType biasTy =
//   op.getOperand(0).getType().cast<Torch::BaseTensorType>();
//   Torch::BaseTensorType inputTy =
//   op.getOperand(1).getType().cast<Torch::BaseTensorType>();
//   Torch::BaseTensorType weightTy =
//   op.getOperand(2).getType().cast<Torch::BaseTensorType>();

//   uint64_t num_output_neurons = resultTy.getSizes()[1];
//   uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);

//   // Use the weight tensor to find the number of input neurons
//   uint64_t num_input_neurons = weightTy.getSizes()[0];
//   uint64_t total_MACs = ofm_volume * num_input_neurons;
//   uint64_t weight_volume = xilinx::xten::getTensorVolume(weightTy);

//   uint64_t ifm_volume = xilinx::xten::getTensorVolume(inputTy);

//   toReturn["ops:MAC"] = total_MACs;
//   toReturn["ops:+"] = ofm_volume;   // Should be gated on whether there is
//   bias at all toReturn["operand:1:activation_in"] = ifm_volume;
//   toReturn["result:0:activation_out"] = ofm_volume;
//   toReturn["operand:0:parameters_in:bias"] =
//   xilinx::xten::getTensorVolume(biasTy);
//   toReturn["operand:2:parameters_in:weight"] = weight_volume;

//   toReturn["reads"] = ifm_volume + weight_volume + num_output_neurons;
//   toReturn["writes"] = ofm_volume;

//   return toReturn;
// }

// as_strided can be zero overhead
// template<>
// std::map<std::string, uint64_t> getStatistics(Torch::AsStridedOp op) {
//   std::map<std::string, uint64_t> toReturn;
//   toReturn["reads"] = 0;
//   toReturn["writes"] = 0;
//   toReturn["operand:0:activation_in"] = 0;
//   toReturn["result:0:activation_out"] = 0;
//   return toReturn;
// }

// batch_norm
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenBatchNormOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  uint64_t op_volume = xilinx::xten::getTensorVolume(resultTy);
  uint64_t weight_volume =
      xilinx::xten::getTensorVolume(op.getOperand(1).getType());
  uint64_t bias_volume =
      xilinx::xten::getTensorVolume(op.getOperand(2).getType());
  toReturn["operand:0:activation_in"] = op_volume;
  toReturn["result:0:activation_out"] = op_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares
  uint64_t ifm_depth = resultTy.getSizes()[1];

  toReturn["ops:+"] = op_volume;  // Add up for mean
  toReturn["ops:*"] = op_volume;  // Square for variance
  toReturn["ops:+"] += op_volume; // Add up squares for variance

  toReturn["ops:*"] += ifm_depth; // Calc channel means
  toReturn["ops:-"] += ifm_depth; // Calc channel vars
  toReturn["ops:*"] += ifm_depth; // Calc channel vars

  toReturn["ops:sqrt"] = ifm_depth; // Convert to SD
  toReturn["ops:/"] = ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume; // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume; // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume; // Bias
  toReturn["ops:*"] += op_volume; // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// _convolution
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenConv2dOp op) {
  return getConv2dStatistics<Torch::AtenConv2dOp>(op);
}

uint64_t getOperandTransferVolume(Torch::AtenConv2dOp op, unsigned int idx,
                                  bool read) {
  return getConv2dOperandTransferVolume<Torch::AtenConv2dOp>(op, idx, read);
}

uint64_t getResultTransferVolume(Torch::AtenConv2dOp op, unsigned int idx,
                                 bool write) {
  return getConv2dResultTransferVolume<Torch::AtenConv2dOp>(op, idx, write);
}

// _convolution_backward
// template<>
// std::map<std::string, uint64_t> getStatistics(ConvolutionBackwardOp op) {

//   std::map<std::string, uint64_t> toReturn;
//   Torch::BaseTensorType dx_out_resultTy =
//   op.getResult(0).getType().cast<Torch::BaseTensorType>(); uint64_t
//   dx_out_volume = xilinx::xten::getTensorVolume(dx_out_resultTy);

//   Torch::BaseTensorType weightTy =
//   op.getOperand(2).getType().cast<Torch::BaseTensorType>(); uint64_t
//   weight_volume = xilinx::xten::getTensorVolume(weightTy); uint64_t
//   loss_in_depth = weightTy.getSizes()[0]; uint64_t kernel_width =
//   weightTy.getSizes()[2]; uint64_t kernel_height = weightTy.getSizes()[3];

//   uint64_t groups = 1; // todo: get this in the same way as the forward path
//   uint64_t MACs_per_loss = (loss_in_depth/groups) * kernel_height *
//   kernel_width;

//   uint64_t total_MACs = dx_out_volume * MACs_per_loss;

//   Torch::BaseTensorType ifmTy =
//   op.getOperand(1).getType().cast<Torch::BaseTensorType>(); uint64_t
//   ifm_volume = xilinx::xten::getTensorVolume(ifmTy); auto ifm_shape =
//   ifmTy.getSizes();

//   uint64_t ifm_bwh = ifm_shape[0]*ifm_shape[2]*ifm_shape[3];  // Batch *
//   height * width: the depth is in the weight shape already total_MACs +=
//   ifm_bwh * weight_volume;

//   Torch::BaseTensorType dx_inTy =
//   op.getOperand(0).getType().cast<Torch::BaseTensorType>(); uint64_t
//   dx_in_volume = xilinx::xten::getTensorVolume(dx_inTy); toReturn["ops:+"] =
//   dx_in_volume;

//   // Reads: Conv_backward reads 3 tensors: the loss in, the activation in and
//   the transposed weights toReturn["reads"] = dx_in_volume + ifm_volume +
//   weight_volume;

//   // Writes: Conv_backward writes 3 tensors: the loss out, gradients for the
//   weights, and gradients for the biases Torch::BaseTensorType biasTy =
//   op.getResult(2).getType().cast<Torch::BaseTensorType>(); uint64_t
//   bias_volume = xilinx::xten::getTensorVolume(biasTy); toReturn["writes"] =
//   dx_out_volume + weight_volume + bias_volume;

//   toReturn["ops:MAC"] = total_MACs;
//   toReturn["operand:0:activation_in"] = dx_in_volume;
//   toReturn["operand:1:activation_in"] = ifm_volume;
//   toReturn["operand:2:parameters_in:weight"] = weight_volume;

//   toReturn["result:0:grad:dx"] = dx_out_volume;
//   toReturn["result:1:grad:dw"] = weight_volume;
//   toReturn["result:2:grad:db"] = bias_volume;

//   return toReturn;
// }

// div
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenDivTensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;

  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// div_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenDiv_TensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// expand can be zero overhead
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenExpandOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// flatten can be zero overhead
template <>
std::map<std::string, uint64_t>
getStatistics(Torch::AtenFlattenUsingIntsOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// tanh
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenTanhOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType inputTy =
      op.getOperand().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();

  uint64_t in_volume = xilinx::xten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// tanh_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenTanh_Op op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType inputTy =
      op.getOperand().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();

  uint64_t in_volume = xilinx::xten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// max_pool2d
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenMaxPool2dOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType inputType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  toReturn["result:0:activation_out"] = ofm_volume;

  uint64_t ifm_volume = xilinx::xten::getTensorVolume(inputType);
  toReturn["input:0:activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  SmallVector<int64_t, 2> kernel_size;
  matchPattern(op.getOperand(1), Torch::m_TorchListOfConstantInts(kernel_size));

  uint64_t aperture = kernel_size[0] * kernel_size[1];
  toReturn["ops:>"] = ofm_volume * (aperture - 1);

  toReturn["reads"] = ifm_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// max_pool2d_with_indices
// template<>
// std::map<std::string, uint64_t> getStatistics(MaxPool2dWithIndicesOp op) {

//     std::map<std::string, uint64_t> toReturn;

//     uint64_t ofm_volume =
//     xilinx::xten::getTensorVolume(op.getResult(0).getType().cast<Torch::BaseTensorType>());
//     uint64_t indices_volume =
//     xilinx::xten::getTensorVolume(op.getResult(1).getType().cast<Torch::BaseTensorType>());

//     toReturn["writes"] = ofm_volume + indices_volume;
//     toReturn["result:0:activation_out"] = ofm_volume;
//     toReturn["result:1:indices_out"] = indices_volume;

//     uint64_t ifm_volume =
//     xilinx::xten::getTensorVolume(op.getOperand(0).getType().cast<Torch::BaseTensorType>());
//     toReturn["reads"] = ifm_volume;
//     toReturn["operand:0:activation_in"] = ifm_volume;

//     // To find the number of compares, we need the filter extent
//     std::vector<int64_t> kernel_size;

//     matchPattern(op.getOperand(1), m_TorchConstantIntList(kernel_size));

//     uint64_t aperture = kernel_size[0] * kernel_size[1];
//     toReturn["ops:>"] = ofm_volume * (aperture-1);

//   return toReturn;
// }

// max_pool2d_with_indicies_backward
// template<>
// std::map<std::string, uint64_t> getStatistics(MaxPool2dWithIndicesBackwardOp
// op) {

//   std::map<std::string, uint64_t> toReturn;

//   Type resultTy = op.getResult().getType();
//   Torch::BaseTensorType tensorResultTy =
//   resultTy.cast<Torch::BaseTensorType>(); uint64_t loss_out_volume =
//   xilinx::xten::getTensorVolume(tensorResultTy); toReturn["writes"] =
//   loss_out_volume;

//   uint64_t loss_in_volume =
//   xilinx::xten::getTensorVolume(op.getOperand(0).getType().cast<Torch::BaseTensorType>());
//   uint64_t act_in_volume  =
//   xilinx::xten::getTensorVolume(op.getOperand(1).getType().cast<Torch::BaseTensorType>());
//   // TODO: Why is this needed? uint64_t indices_volume  =
//   xilinx::xten::getTensorVolume(op.getOperand(7).getType().cast<Torch::BaseTensorType>());
//   toReturn["reads"] = loss_in_volume + act_in_volume + indices_volume;
//   toReturn["operand:0:activation_in"] = loss_in_volume;
//   toReturn["operand:1:activation_in"] = act_in_volume;
//   toReturn["operand:3:activation_in"] = indices_volume;
//   toReturn["result:0:grad:dx"] = loss_out_volume;

//   return toReturn;
// }

// mean
// template<>
// std::map<std::string, uint64_t> getStatistics(MeanOp op) {

//   std::map<std::string, uint64_t> toReturn;

//   Torch::BaseTensorType resultTy =
//   op.getResult().getType().cast<Torch::BaseTensorType>();
//   Torch::BaseTensorType aType =
//   op.getOperand().getType().cast<Torch::BaseTensorType>();

//   uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
//   toReturn["ops:+"] = ofm_volume;
//   toReturn["result:0:activation_out"] = ofm_volume;

//   // Find the size of the A and B operands
//   uint64_t a_volume = xilinx::xten::getTensorVolume(aType);

//   toReturn["operand:0:activation_in"] = a_volume;

//   toReturn["reads"] = a_volume;
//   toReturn["writes"] = ofm_volume;

//   return toReturn;
// }

// mm
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenMmOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  Torch::BaseTensorType weightTy =
      op.getOperand(1).getType().cast<Torch::BaseTensorType>();
  uint64_t num_input_neurons = weightTy.getSizes()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  toReturn["ops:MAC"] = total_MACs;

  uint64_t loss_in_volume = xilinx::xten::getTensorVolume(
      op.getOperand(0).getType().cast<Torch::BaseTensorType>());
  uint64_t weight_volume = xilinx::xten::getTensorVolume(
      op.getOperand(1).getType().cast<Torch::BaseTensorType>());
  toReturn["reads"] = loss_in_volume + weight_volume;
  toReturn["writes"] = ofm_volume;

  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = weight_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  return toReturn;
}

// mul
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenMulTensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mul_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenMul_TensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// batchnorm backward
// template<>
// std::map<std::string, uint64_t> getStatistics(NativeBatchNormBackwardOp op) {

//   std::map<std::string, uint64_t> toReturn;

//   ShapedType inputTy = op.getOperand(0).getType().cast<ShapedType>();
//   uint64_t input_volume = xilinx::xten::getTensorVolume(inputTy);
//   uint64_t input_channels = inputTy.getSizes()[1];

//   // from
//   https://gitenterprise.xilinx.com/nfraser/torchscope/blob/master/torchscope/helper.py
//   // # 3 components make up the gradInput: 1 gradInput, 2 gradMean, 3 gradVar
//   // # totalGradInput = gradInput + (dL / dMean * dMean / dInput) +
//   // #                  (dL / dVar * dVar / dInput)

//   // # gradInput
//   // total_ops["backward"]["*"] = in_c * (in_h*in_w*batch_size) # scale
//   // # Bootstrap from previous
//   // #total_ops["backward"]["sqrt"] = in_c # Convert to std_dev
//   // #total_ops["backward"]["/"] = in_c # Calculate inverse sqrt first
//   toReturn["ops:*"] = input_volume; // scale

//   // # dL / dGradVar
//   // total_ops["backward"]["pow"] = in_c
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c
//   // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
//   in_h*in_w*batch_size # Subtract mean, bootstrap from previous calculation
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c *
//   (in_h*in_w*batch_size) toReturn["ops:pow"] = input_channels;;
//   toReturn["ops:*"] += input_channels;
//   toReturn["ops:*"] += input_volume;

//   // # dL / dGradMean
//   // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
//   (in_h*in_w*batch_size) # bootstrap from previous
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale
//   gradMean
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # eltwise
//   with dL / dGradVar
//   // total_ops["backward"]["+"] = in_c * (in_h*in_w*batch_size) # sum
//   gradXhat
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale
//   gradXhat toReturn["ops:*"] += input_channels; // scale gradMean
//   toReturn["ops:*"] += input_channels; // eltwise with dL / dGradVar
//   toReturn["ops:+"] = input_volume; // sum gradXhat
//   toReturn["ops:*"] += input_channels; // scale gradXhat

//   // # totalGradInput
//   // total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c *
//   (in_h*in_w*batch_size) # Subtract mean, can't bootstrap this one
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale
//   dL / dMean
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale
//   dL / dVar
//   // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c *
//   (in_h*in_w*batch_size) # Eltwise multiply by dL / dVar
//   // total_ops["backward"]["+"] = total_ops["backward"]["+"] + 2 * in_c *
//   (in_h*in_w*batch_size) # Accumulate gradient terms toReturn["ops:+"] +=
//   input_volume; // Subtract mean, can't bootstrap this one toReturn["ops:*"]
//   += input_channels; // scale dL / dMean toReturn["ops:*"] += input_channels;
//   // scale dL / dVar toReturn["ops:*"] += input_volume; // Eltwise multiply
//   by dL / dVar toReturn["OPS:+"] += 2 * input_volume; // Accumulate gradient
//   terms

//   uint64_t reads = 0;
//   for (int i=0; i<7; i++) {
//     auto v = xilinx::xten::getTensorVolume(op.getOperand(i).getType());
//     toReturn["operand:"+std::to_string(i)+":activation_in"] = v;
//     reads += v;
//   }

//   uint64_t writes = 0;
//   for (int i=0; i<3; i++) {
//     auto v = xilinx::xten::getTensorVolume(op.getResult(i).getType());
//     toReturn["result:"+std::to_string(i)+":grad"] = v;
//     writes += v;
//   }

//   toReturn["reads"] = reads;
//   toReturn["writes"] = writes;

//   return toReturn;
// }

// relu
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenReluOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType inputTy =
      op.getOperand().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();

  uint64_t in_volume = xilinx::xten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// relu_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenRelu_Op op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType inputTy =
      op.getOperand().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();

  uint64_t in_volume = xilinx::xten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// sub
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenSubTensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sub_
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenSub_TensorOp op) {

  std::map<std::string, uint64_t> toReturn;

  Torch::BaseTensorType resultTy =
      op.getResult().getType().cast<Torch::BaseTensorType>();
  Torch::BaseTensorType aType =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::xten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::xten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sum
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenSumOp op) {

  std::map<std::string, uint64_t> toReturn;
  Torch::BaseTensorType ty =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  uint64_t volume = xilinx::xten::getTensorVolume(ty);

  toReturn["ops:+"] = volume;

  toReturn["operand:0:activation_in"] = volume;
  toReturn["result:0:activation_out"] = volume;

  toReturn["reads"] = volume;
  toReturn["writes"] = volume;

  return toReturn;
}

// scalar mul
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenMulScalarOp op) {

  std::map<std::string, uint64_t> toReturn;
  Torch::BaseTensorType ty =
      op.getOperand(0).getType().cast<Torch::BaseTensorType>();
  uint64_t volume = xilinx::xten::getTensorVolume(ty);

  toReturn["ops:*"] = volume;

  toReturn["operand:0:activation_in"] = volume;
  toReturn["result:0:activation_out"] = volume;

  toReturn["reads"] = volume;
  toReturn["writes"] = volume;

  return toReturn;
}

// threshold_backward
// template<>
// std::map<std::string, uint64_t> getStatistics(ThresholdBackwardOp op) {

//   std::map<std::string, uint64_t> toReturn;
//   uint64_t loss_in_volume =
//   xilinx::xten::getTensorVolume(op.getOperand(0).getType().cast<Torch::BaseTensorType>());
//   uint64_t act_in_volume  =
//   xilinx::xten::getTensorVolume(op.getOperand(1).getType().cast<Torch::BaseTensorType>());
//   uint64_t loss_out_volume =
//   xilinx::xten::getTensorVolume(op.getResult().getType().cast<Torch::BaseTensorType>());

//   toReturn["reads"]  = toReturn["operand:0:activation_in"] = loss_in_volume +
//   act_in_volume; toReturn["writes"] = toReturn["result:0:grad:dx"] =
//   loss_out_volume;

//   return toReturn;
// }

// transpose can be zero overhead
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenTransposeIntOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// unsqueeze can be zero overhead
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenUnsqueezeOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// view can be zero overhead
template <>
std::map<std::string, uint64_t> getStatistics(Torch::AtenViewOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

std::map<std::string, uint64_t> getATenOpStats(Operation *op) {

#define GET_STATS(T)                                                           \
  if (isa<T>(op))                                                              \
    return getStatistics<T>(cast<T>(op));
  GET_STATS(Torch::AtenMulScalarOp)
  GET_STATS(Torch::AtenAddTensorOp)
  GET_STATS(Torch::AtenAdd_TensorOp)
  //  GET_STATS(AddmmOp)
  //  GET_STATS(AsStridedOp)
  GET_STATS(Torch::AtenBatchNormOp)
  GET_STATS(Torch::AtenConv2dOp)
  //  GET_STATS(ConvolutionBackwardOp)
  GET_STATS(Torch::AtenDivTensorOp)
  GET_STATS(Torch::AtenDiv_TensorOp)
  GET_STATS(Torch::AtenExpandOp)
  GET_STATS(Torch::AtenFlattenUsingIntsOp)
  GET_STATS(Torch::AtenTanhOp)
  GET_STATS(Torch::AtenTanh_Op)
  GET_STATS(Torch::AtenMaxPool2dOp)
  //  GET_STATS(MaxPool2dWithIndicesOp)
  //  GET_STATS(MaxPool2dWithIndicesBackwardOp)
  //  GET_STATS(MeanOp)
  GET_STATS(Torch::AtenMmOp)
  GET_STATS(Torch::AtenMulTensorOp)
  GET_STATS(Torch::AtenMul_TensorOp)
  //  GET_STATS(NativeBatchNormBackwardOp)
  GET_STATS(Torch::AtenReluOp)
  GET_STATS(Torch::AtenRelu_Op)
  GET_STATS(Torch::AtenSubTensorOp)
  GET_STATS(Torch::AtenSub_TensorOp)
  GET_STATS(Torch::AtenSumOp)
  //  GET_STATS(ThresholdBackwardOp)
  GET_STATS(Torch::AtenTransposeIntOp)
  GET_STATS(Torch::AtenUnsqueezeOp)
  GET_STATS(Torch::AtenViewOp)

  return std::map<std::string, uint64_t>();
}

} // namespace xten
} // namespace xilinx
#endif
