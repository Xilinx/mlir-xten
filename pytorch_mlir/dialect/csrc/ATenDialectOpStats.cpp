#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"
#include "Util.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>

#define DEBUG_TYPE "aten-op-stats"

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;
using namespace xilinx;

namespace {

std::vector<uint64_t> unpackListConstant(Value op) {
  std::vector<uint64_t> v;
  auto co = cast<NPCOMP::aten::ConstantOp>(op.getDefiningOp());
  DenseElementsAttr a = co->getAttrOfType<DenseElementsAttr>("value");
  for (auto i : a.getIntValues())
    v.push_back(i.getSExtValue());
  return v;
};

template<class T>
std::map<std::string, uint64_t> getConv2dStatistics(T *o) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = o->getResult().getType().template cast<TensorType>();
  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  TensorType biasTy = o->bias().getType().template cast<TensorType>();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  //uint64_t ofm_depth = resultTy.getShape()[1];

  uint64_t ifm_depth = inputTy.getShape()[1];
  uint64_t kernel_height = weightTy.getShape()[2];
  uint64_t kernel_width = weightTy.getShape()[3];

  auto co = cast<NPCOMP::aten::ConstantOp>(o->groups().getDefiningOp());
  auto ia = co->template getAttrOfType<IntegerAttr>("value");
  uint64_t groups = ia.getValue().getZExtValue();

  // Number of forward MACs per pixel =
  //  kernel_width * kernel_height * ifm_depth / groups
  uint64_t MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
  uint64_t total_MACs = ofm_volume * MACs_per_OFM;

  uint64_t ifm_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t weight_volume = xilinx::air::getTensorVolume(weightTy);
  uint64_t bias_volume = xilinx::air::getTensorVolume(biasTy);

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

template<class T>
uint64_t getConv2dOperandTransferVolume(T *o, unsigned int idx, bool read) {

  if (!read) return 0;

  double vol = xilinx::air::getTensorVolume(o->getOperand(idx).getType());
  if (simple_conv2d_model)
    return vol;

  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  TensorType resultTy = o->getResult().getType().template cast<TensorType>();

  float filter_width = weightTy.getShape()[2];
  float filter_height = weightTy.getShape()[3];

  float batch_sw = inputTy.getShape()[0];
  //float ifm_depth_sw = inputTy.getShape()[1];
  float ih = inputTy.getShape()[2];
  //float iw = inputTy.getShape()[3];

  float ofm_depth_sw = resultTy.getShape()[1];

  const float batch_hw = 4;
  //const float ifm_depth_hw = 32;
  const float ofm_depth_hw = 32;

  const float ifm_tile_height = 4;
  //const float ifm_tile_width = 4;
  //const float ofm_tile_height = 4;
  //const float ofm_tile_width = 4;

  float ifm_aperture = ifm_tile_height - ceilf(filter_height/2.0f);
  float ifm_overlap = ceilf(filter_height/2.0f);

  float bl = ceilf(batch_sw / batch_hw);
  float ol = ceilf(ofm_depth_sw / ofm_depth_hw);
  //float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float ifm_overhead = 1.0f;
  float weight_overhead = 1.0f;
  if (filter_width > 1) {
    ifm_overhead = ol * ifm_tile_height * ((ih - ifm_overlap) / (ih * ifm_aperture));
    weight_overhead = bl;
  }
  else {
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

template<class T>
uint64_t getConv2dResultTransferVolume(T *o, unsigned int idx, bool write) {

  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType resultTy = o->getResult().getType().template cast<TensorType>();

  if (simple_conv2d_model) {
    if (write)
      return xilinx::air::getTensorVolume(resultTy);
    else
      return 0;
  }

  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  float filter_width = weightTy.getShape()[2];
  //float filter_height = weightTy.getShape()[3];

  float ifm_depth_sw = inputTy.getShape()[1];
  const float ifm_depth_hw = 32;

  float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float write_output_overhead = 1.0f;
  float read_output_cost = 1.0f;
  
  if (filter_width > 1) {
    write_output_overhead = il;
    read_output_cost = il;
  }

  double vol = xilinx::air::getTensorVolume(resultTy);

  if (write) {
    LLVM_DEBUG(llvm::outs() << "write_output_overhead:" << write_output_overhead << "\n");
    return vol * write_output_overhead;
  } else {
    LLVM_DEBUG(llvm::outs() << "read_output_cost:" << read_output_cost << "\n");
    return vol * read_output_cost;
  }
}

} // namespace

namespace xilinx {
namespace air {

// acap conv2d bn relu

std::map<std::string, uint64_t> Conv2dBatchNormReLUOp::getStatistics() {
  return getConv2dStatistics<Conv2dBatchNormReLUOp>(this);
}

uint64_t Conv2dBatchNormReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dBatchNormReLUOp>(this, idx, read);
}

uint64_t Conv2dBatchNormReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dBatchNormReLUOp>(this, idx, write);
}

// acap conv2d relu

std::map<std::string, uint64_t> Conv2dReLUOp::getStatistics() {
  return getConv2dStatistics<Conv2dReLUOp>(this);
}

uint64_t Conv2dReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dReLUOp>(this, idx, read);
}

uint64_t Conv2dReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dReLUOp>(this, idx, write);
}

// acap conv2d

std::map<std::string, uint64_t> Conv2dOp::getStatistics() {
  return getConv2dStatistics<Conv2dOp>(this);
}

uint64_t Conv2dOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dOp>(this, idx, read);
}

uint64_t Conv2dOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dOp>(this, idx, write);
}

}
}

#if 0

namespace xilinx {
namespace aten {

// add
std::map<std::string, uint64_t> AddOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// add_
std::map<std::string, uint64_t> AddUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// addmm
std::map<std::string, uint64_t> AddmmOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  // For linear, we need the number of output neurons and the number of input neurons
  // Then the number of forward MACs is input * output
  // And the number of adds is output if there is bias

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType biasTy = getOperand(0).getType().cast<TensorType>();
  TensorType inputTy = getOperand(1).getType().cast<TensorType>();
  TensorType weightTy = getOperand(2).getType().cast<TensorType>();

  uint64_t num_output_neurons = resultTy.getShape()[1];
  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  uint64_t weight_volume = xilinx::air::getTensorVolume(weightTy);

  uint64_t ifm_volume = xilinx::air::getTensorVolume(inputTy);

  toReturn["ops:MAC"] = total_MACs;
  toReturn["ops:+"] = ofm_volume;   // Should be gated on whether there is bias at all
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:0:parameters_in:bias"] = xilinx::air::getTensorVolume(biasTy);
  toReturn["operand:2:parameters_in:weight"] = weight_volume;

  toReturn["reads"] = ifm_volume + weight_volume + num_output_neurons;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// as_strided can be zero overhead
std::map<std::string, uint64_t> AsStridedOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = 0;
  toReturn["writes"] = 0;
  toReturn["operand:0:activation_in"] = 0;
  toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// batch_norm
std::map<std::string, uint64_t> BatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = xilinx::air::getTensorVolume(resultTy);
  uint64_t weight_volume = xilinx::air::getTensorVolume(getOperand(1).getType());
  uint64_t bias_volume = xilinx::air::getTensorVolume(getOperand(2).getType());
  toReturn["operand:0:activation_in"] = op_volume;
  toReturn["result:0:activation_out"] = op_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares
  uint64_t ifm_depth = resultTy.getShape()[1];

  toReturn["ops:+"] = op_volume;   // Add up for mean
  toReturn["ops:*"] = op_volume;   // Square for variance
  toReturn["ops:+"] += op_volume;  // Add up squares for variance

  toReturn["ops:*"] += ifm_depth;   // Calc channel means
  toReturn["ops:-"] += ifm_depth;   // Calc channel vars
  toReturn["ops:*"] += ifm_depth;   // Calc channel vars

  toReturn["ops:sqrt"] = ifm_depth;  // Convert to SD
  toReturn["ops:/"] = ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume;   // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume;   // Bias
  toReturn["ops:*"] += op_volume;   // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// _convolution
std::map<std::string, uint64_t> ConvolutionOp::getStatistics() {
  return getConv2dStatistics<ConvolutionOp>(this);
}

uint64_t ConvolutionOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<ConvolutionOp>(this, idx, read);
}

uint64_t ConvolutionOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<ConvolutionOp>(this, idx, write);
}

// _convolution_backward
std::map<std::string, uint64_t> ConvolutionBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  TensorType dx_out_resultTy = getResult(0).getType().cast<TensorType>();
  uint64_t dx_out_volume = xilinx::air::getTensorVolume(dx_out_resultTy);

  TensorType weightTy = getOperand(2).getType().cast<TensorType>();
  uint64_t weight_volume = xilinx::air::getTensorVolume(weightTy);
  uint64_t loss_in_depth = weightTy.getShape()[0];
  uint64_t kernel_width = weightTy.getShape()[2];
  uint64_t kernel_height = weightTy.getShape()[3];

  uint64_t groups = 1; // todo: get this in the same way as the forward path
  uint64_t MACs_per_loss = (loss_in_depth/groups) * kernel_height * kernel_width;

  uint64_t total_MACs = dx_out_volume * MACs_per_loss;

  TensorType ifmTy = getOperand(1).getType().cast<TensorType>();
  uint64_t ifm_volume = xilinx::air::getTensorVolume(ifmTy);
  auto ifm_shape = ifmTy.getShape();

  uint64_t ifm_bwh = ifm_shape[0]*ifm_shape[2]*ifm_shape[3];  // Batch * height * width: the depth is in the weight shape already
  total_MACs += ifm_bwh * weight_volume;

  TensorType dx_inTy = getOperand(0).getType().cast<TensorType>();
  uint64_t dx_in_volume = xilinx::air::getTensorVolume(dx_inTy);
  toReturn["ops:+"] = dx_in_volume;

  // Reads: Conv_backward reads 3 tensors: the loss in, the activation in and the transposed weights
  toReturn["reads"] = dx_in_volume + ifm_volume + weight_volume;

  // Writes: Conv_backward writes 3 tensors: the loss out, gradients for the weights, and gradients for the biases
  TensorType biasTy = getResult(2).getType().cast<TensorType>();
  uint64_t bias_volume = xilinx::air::getTensorVolume(biasTy);
  toReturn["writes"] = dx_out_volume + weight_volume + bias_volume;

  toReturn["ops:MAC"] = total_MACs;
  toReturn["operand:0:activation_in"] = dx_in_volume;
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["operand:2:parameters_in:weight"] = weight_volume;

  toReturn["result:0:grad:dx"] = dx_out_volume;
  toReturn["result:1:grad:dw"] = weight_volume;
  toReturn["result:2:grad:db"] = bias_volume;
 
  return toReturn;
}

// div
std::map<std::string, uint64_t> DivOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;

  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// div_
std::map<std::string, uint64_t> DivUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// expand can be zero overhead
std::map<std::string, uint64_t> ExpandOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// flatten can be zero overhead
std::map<std::string, uint64_t> FlattenOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// hardtanh
std::map<std::string, uint64_t> HardtanhOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// hardtanh_
std::map<std::string, uint64_t> HardtanhUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// max_pool2d
std::map<std::string, uint64_t> MaxPool2dOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType inputType = getOperand(0).getType().cast<TensorType>();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["result:0:activation_out"] = ofm_volume;

  uint64_t ifm_volume = xilinx::air::getTensorVolume(inputType);
  toReturn["input:0:activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel_size = unpackListConstant(getOperand(1));

  uint64_t aperture = kernel_size[0] * kernel_size[1];
  toReturn["ops:>"] = ofm_volume * (aperture-1);

  toReturn["reads"] = ifm_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// max_pool2d_with_indices
std::map<std::string, uint64_t> MaxPool2dWithIndicesOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  uint64_t ofm_volume = xilinx::air::getTensorVolume(getResult(0).getType().cast<TensorType>());
  uint64_t indices_volume = xilinx::air::getTensorVolume(getResult(1).getType().cast<TensorType>());

  toReturn["writes"] = ofm_volume + indices_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["result:1:indices_out"] = indices_volume;

  uint64_t ifm_volume = xilinx::air::getTensorVolume(getOperand(0).getType().cast<TensorType>());
  toReturn["reads"] = ifm_volume;
  toReturn["operand:0:activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel_size = unpackListConstant(getOperand(1));

  uint64_t aperture = kernel_size[0] * kernel_size[1];
  toReturn["ops:>"] = ofm_volume * (aperture-1);

  return toReturn;
}

// max_pool2d_with_indicies_backward
std::map<std::string, uint64_t> MaxPool2dWithIndicesBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult().getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t loss_out_volume = xilinx::air::getTensorVolume(tensorResultTy);
  toReturn["writes"] = loss_out_volume;

  uint64_t loss_in_volume = xilinx::air::getTensorVolume(getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume  = xilinx::air::getTensorVolume(getOperand(1).getType().cast<TensorType>()); // TODO: Why is this needed?
  uint64_t indices_volume  = xilinx::air::getTensorVolume(getOperand(7).getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + act_in_volume + indices_volume;
  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = act_in_volume;
  toReturn["operand:3:activation_in"] = indices_volume;
  toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// mean
std::map<std::string, uint64_t> MeanOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand().getType().cast<TensorType>();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);

  toReturn["operand:0:activation_in"] = a_volume;

  toReturn["reads"] = a_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mm
std::map<std::string, uint64_t> MMOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  TensorType weightTy = getOperand(1).getType().cast<TensorType>();
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  toReturn["ops:MAC"] = total_MACs;

  uint64_t loss_in_volume = xilinx::air::getTensorVolume(getOperand(0).getType().cast<TensorType>());
  uint64_t weight_volume = xilinx::air::getTensorVolume(getOperand(1).getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + weight_volume;
  toReturn["writes"] = ofm_volume;

  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = weight_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  return toReturn;
}

// mul
std::map<std::string, uint64_t> MulOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mul_
std::map<std::string, uint64_t> MulUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// native_batch_norm
std::map<std::string, uint64_t> NativeBatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = xilinx::air::getTensorVolume(resultTy);
  uint64_t weight_volume = xilinx::air::getTensorVolume(getOperand(1).getType());
  uint64_t bias_volume = xilinx::air::getTensorVolume(getOperand(2).getType());
  toReturn["operand:0:activation_in"] = op_volume;
  toReturn["result:0:activation_out"] = op_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares
  uint64_t ifm_depth = resultTy.getShape()[1];

  toReturn["ops:+"] = op_volume;   // Add up for mean
  toReturn["ops:*"] = op_volume;   // Square for variance
  toReturn["ops:+"] += op_volume;  // Add up squares for variance

  toReturn["ops:*"] += ifm_depth;   // Calc channel means
  toReturn["ops:-"] += ifm_depth;   // Calc channel vars
  toReturn["ops:*"] += ifm_depth;   // Calc channel vars

  toReturn["ops:sqrt"] = ifm_depth;  // Convert to SD
  toReturn["ops:/"] = ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume;   // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume;   // Bias
  toReturn["ops:*"] += op_volume;   // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// batchnorm backward
std::map<std::string, uint64_t> NativeBatchNormBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  ShapedType inputTy = getOperand(0).getType().cast<ShapedType>();
  uint64_t input_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t input_channels = inputTy.getShape()[1];

  // from https://gitenterprise.xilinx.com/nfraser/torchscope/blob/master/torchscope/helper.py
  // # 3 components make up the gradInput: 1 gradInput, 2 gradMean, 3 gradVar
  // # totalGradInput = gradInput + (dL / dMean * dMean / dInput) +
  // #                  (dL / dVar * dVar / dInput)

  // # gradInput
  // total_ops["backward"]["*"] = in_c * (in_h*in_w*batch_size) # scale
  // # Bootstrap from previous
  // #total_ops["backward"]["sqrt"] = in_c # Convert to std_dev
  // #total_ops["backward"]["/"] = in_c # Calculate inverse sqrt first
  toReturn["ops:*"] = input_volume; // scale

  // # dL / dGradVar
  // total_ops["backward"]["pow"] = in_c
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c
  // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c * in_h*in_w*batch_size # Subtract mean, bootstrap from previous calculation
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c * (in_h*in_w*batch_size)
  toReturn["ops:pow"] = input_channels;;
  toReturn["ops:*"] += input_channels;
  toReturn["ops:*"] += input_volume;

  // # dL / dGradMean
  // #total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c * (in_h*in_w*batch_size) # bootstrap from previous
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale gradMean
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # eltwise with dL / dGradVar
  // total_ops["backward"]["+"] = in_c * (in_h*in_w*batch_size) # sum gradXhat
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale gradXhat
  toReturn["ops:*"] += input_channels; // scale gradMean
  toReturn["ops:*"] += input_channels; // eltwise with dL / dGradVar
  toReturn["ops:+"] = input_volume; // sum gradXhat
  toReturn["ops:*"] += input_channels; // scale gradXhat

  // # totalGradInput
  // total_ops["backward"]["+"] = total_ops["backward"]["+"] + in_c * (in_h*in_w*batch_size) # Subtract mean, can't bootstrap this one
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale dL / dMean
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c # scale dL / dVar
  // total_ops["backward"]["*"] = total_ops["backward"]["*"] + in_c * (in_h*in_w*batch_size) # Eltwise multiply by dL / dVar
  // total_ops["backward"]["+"] = total_ops["backward"]["+"] + 2 * in_c * (in_h*in_w*batch_size) # Accumulate gradient terms
  toReturn["ops:+"] += input_volume; // Subtract mean, can't bootstrap this one
  toReturn["ops:*"] += input_channels; // scale dL / dMean
  toReturn["ops:*"] += input_channels; // scale dL / dVar
  toReturn["ops:*"] += input_volume; // Eltwise multiply by dL / dVar
  toReturn["OPS:+"] += 2 * input_volume; // Accumulate gradient terms

  uint64_t reads = 0;
  for (int i=0; i<7; i++) {
    auto v = xilinx::air::getTensorVolume(getOperand(i).getType());
    toReturn["operand:"+std::to_string(i)+":activation_in"] = v;
    reads += v;
  }

  uint64_t writes = 0;
  for (int i=0; i<3; i++) {
    auto v = xilinx::air::getTensorVolume(getResult(i).getType());
    toReturn["result:"+std::to_string(i)+":grad"] = v;
    writes += v;
  }

  toReturn["reads"] = reads;
  toReturn["writes"] = writes;

  return toReturn;
}

// relu
std::map<std::string, uint64_t> ReLUOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand().getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// relu_
std::map<std::string, uint64_t> ReLUUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand().getType().cast<TensorType>();
  TensorType resultTy = getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::air::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// sub
std::map<std::string, uint64_t> SubOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// sub_
std::map<std::string, uint64_t> SubUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult().getType().cast<TensorType>();
  TensorType aType = getOperand(0).getType().cast<TensorType>();
  Type bType = getOperand(1).getType();

  uint64_t ofm_volume = xilinx::air::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::air::getTensorVolume(aType);
  uint64_t b_volume = xilinx::air::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sum
std::map<std::string, uint64_t> SumOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  TensorType ty = getOperand(0).getType().cast<TensorType>();
  uint64_t volume = xilinx::air::getTensorVolume(ty);

  toReturn["ops:+"] = volume;

  toReturn["operand:0:activation_in"] = volume;
  toReturn["result:0:activation_out"] = volume;

  toReturn["reads"] = volume;
  toReturn["writes"] = volume;

  return toReturn;
}

// threshold_backward
std::map<std::string, uint64_t> ThresholdBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  uint64_t loss_in_volume = xilinx::air::getTensorVolume(getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume  = xilinx::air::getTensorVolume(getOperand(1).getType().cast<TensorType>());
  uint64_t loss_out_volume = xilinx::air::getTensorVolume(getResult().getType().cast<TensorType>());

  toReturn["reads"]  = toReturn["operand:0:activation_in"] = loss_in_volume + act_in_volume;
  toReturn["writes"] = toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// transpose can be zero overhead
std::map<std::string, uint64_t> TransposeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// unsqueeze can be zero overhead
std::map<std::string, uint64_t> UnsqueezeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// view can be zero overhead
std::map<std::string, uint64_t> ViewOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

} // namespace aten
} // namespace xilinx
#endif
