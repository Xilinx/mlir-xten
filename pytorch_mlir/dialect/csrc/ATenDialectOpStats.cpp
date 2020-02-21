#include "ATenDialect.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include <iostream>

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;

namespace {

uint64_t getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(const Type ty) {
  if (auto t = ty.dyn_cast<ShapedType>()) {
    return getTensorVolume(t);
  }
  else {
    return 1;
  }
}

std::vector<uint64_t> unpackListConstant(Value *op) {
  std::vector<uint64_t> v;
  auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
  for (auto i : a.getIntValues())
    v.push_back(i.getSExtValue());
  return v;
};

} // namespace

namespace xilinx {
namespace aten {

// add
std::map<std::string, uint64_t> AddOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// add_
std::map<std::string, uint64_t> AddUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

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

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType biasTy = getOperand(0)->getType().cast<TensorType>();
  TensorType inputTy = getOperand(1)->getType().cast<TensorType>();
  TensorType weightTy = getOperand(2)->getType().cast<TensorType>();

  uint64_t num_output_neurons = resultTy.getShape()[1];
  uint64_t ofm_volume = getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  uint64_t weight_volume = getTensorVolume(weightTy);

  uint64_t ifm_volume = getTensorVolume(inputTy);

  toReturn["ops:MAC"] = total_MACs;
  toReturn["ops:+"] = ofm_volume;   // Should be gated on whether there is bias at all
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:0:parameters_in:bias"] = getTensorVolume(biasTy);
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
  return toReturn;
}

// batch_norm
std::map<std::string, uint64_t> BatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0)->getType().cast<TensorType>();
  uint64_t op_volume = getTensorVolume(resultTy);
  uint64_t weight_volume = getTensorVolume(getOperand(1)->getType());
  uint64_t bias_volume = getTensorVolume(getOperand(2)->getType());
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
  toReturn["ops:/"] += ifm_depth;    // Get the reciprocal

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

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType inputTy = getOperand(0)->getType().cast<TensorType>();
  TensorType weightTy = getOperand(1)->getType().cast<TensorType>();
  TensorType biasTy = getOperand(2)->getType().cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  uint64_t ofm_depth = resultTy.getShape()[1];

  uint64_t ifm_depth = inputTy.getShape()[1];
  uint64_t kernel_width = weightTy.getShape()[2];
  uint64_t kernel_height = weightTy.getShape()[3];

  auto co = cast<xilinx::aten::ConstantOp>(getOperand(8)->getDefiningOp());
  auto ia = co.getAttrOfType<IntegerAttr>("value");
  uint64_t groups = ia.getValue().getZExtValue();

  // Number of forward MACs per pixel =
  //  kernel_width * kernel_height * ifm_depth / groups
  uint64_t MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
  uint64_t total_MACs = ofm_volume * MACs_per_OFM;

  uint64_t ifm_volume = getTensorVolume(inputTy);
  uint64_t weight_volume = getTensorVolume(weightTy);
  uint64_t bias_volume = getTensorVolume(biasTy);

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

// _convolution_backward
std::map<std::string, uint64_t> ConvolutionBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  TensorType dx_out_resultTy = getResult(0)->getType().cast<TensorType>();
  uint64_t dx_out_volume = getTensorVolume(dx_out_resultTy);

  TensorType weightTy = getOperand(2)->getType().cast<TensorType>();
  uint64_t weight_volume = getTensorVolume(weightTy);
  uint64_t loss_in_depth = weightTy.getShape()[0];
  uint64_t kernel_width = weightTy.getShape()[2];
  uint64_t kernel_height = weightTy.getShape()[3];

  uint64_t groups = 1; // todo: get this in the same way as the forward path
  uint64_t MACs_per_loss = (loss_in_depth/groups) * kernel_height * kernel_width;

  uint64_t total_MACs = dx_out_volume * MACs_per_loss;

  TensorType ifmTy = getOperand(1)->getType().cast<TensorType>();
  uint64_t ifm_volume = getTensorVolume(ifmTy);
  auto ifm_shape = ifmTy.getShape();

  uint64_t ifm_bwh = ifm_shape[0]*ifm_shape[2]*ifm_shape[3];  // Batch * height * width: the depth is in the weight shape already
  total_MACs += ifm_bwh * weight_volume;

  TensorType dx_inTy = getOperand(0)->getType().cast<TensorType>();
  uint64_t dx_in_volume = getTensorVolume(dx_inTy);
  toReturn["ops:+"] = dx_in_volume;

  // Reads: Conv_backward reads 3 tensors: the loss in, the activation in and the transposed weights
  toReturn["reads"] = dx_in_volume + ifm_volume + weight_volume;

  // Writes: Conv_backward writes 3 tensors: the loss out, gradients for the weights, and gradients for the biases
  TensorType biasTy = getResult(2)->getType().cast<TensorType>();
  uint64_t bias_volume = getTensorVolume(biasTy);
  toReturn["writes"] = dx_out_volume + weight_volume + bias_volume; 

  toReturn["ops:MAC"] = total_MACs;
  return toReturn;
}

// div
std::map<std::string, uint64_t> DivOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;

  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// div_
std::map<std::string, uint64_t> DivUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// max_pool2d
std::map<std::string, uint64_t> MaxPool2dOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType inputType = getOperand(0)->getType().cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["result:0:activation_out"] = ofm_volume;

  uint64_t ifm_volume = getTensorVolume(inputType);
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

  uint64_t ofm_volume = getTensorVolume(getResult(0)->getType().cast<TensorType>());
  uint64_t indices_volume = getTensorVolume(getResult(1)->getType().cast<TensorType>());

  toReturn["writes"] = ofm_volume + indices_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["result:1:indices_out"] = indices_volume;

  uint64_t ifm_volume = getTensorVolume(getOperand(0)->getType().cast<TensorType>());
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

  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t loss_out_volume = getTensorVolume(tensorResultTy);
  toReturn["writes"] = loss_out_volume;

  uint64_t loss_in_volume = getTensorVolume(getOperand(0)->getType().cast<TensorType>());
  uint64_t act_in_volume  = getTensorVolume(getOperand(1)->getType().cast<TensorType>()); // TODO: Why is this needed?
  uint64_t indices_volume  = getTensorVolume(getOperand(7)->getType().cast<TensorType>()); 
  toReturn["reads"] = loss_in_volume + act_in_volume + indices_volume;

  return toReturn;
}

// mean
std::map<std::string, uint64_t> MeanOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand()->getType().cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);

  toReturn["operand:0:activation_in"] = a_volume;

  toReturn["reads"] = a_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mm
std::map<std::string, uint64_t> MMOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  uint64_t ofm_volume = getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  TensorType weightTy = getOperand(1)->getType().cast<TensorType>();
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  toReturn["ops:MAC"] = total_MACs;

  uint64_t loss_in_volume = getTensorVolume(getOperand(0)->getType().cast<TensorType>());
  uint64_t weight_volume = getTensorVolume(getOperand(1)->getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + weight_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mul
std::map<std::string, uint64_t> MulOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mul_
std::map<std::string, uint64_t> MulUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// native_batch_norm
std::map<std::string, uint64_t> NativeBatchNormOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult(0)->getType().cast<TensorType>();
  uint64_t op_volume = getTensorVolume(resultTy);
  uint64_t weight_volume = getTensorVolume(getOperand(1)->getType());
  uint64_t bias_volume = getTensorVolume(getOperand(2)->getType());
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
  toReturn["ops:/"] += ifm_depth;    // Get the reciprocal

  toReturn["ops:+"] += op_volume;   // Subtract mean off each pixel
  toReturn["ops:*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["ops:+"] += op_volume;   // Bias
  toReturn["ops:*"] += op_volume;   // Scale

  toReturn["reads"] = op_volume + weight_volume + bias_volume;
  toReturn["writes"] = op_volume;

  return toReturn;
}

// relu
std::map<std::string, uint64_t> ReLUOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = getOperand()->getType().cast<TensorType>();
  TensorType resultTy = getResult()->getType().cast<TensorType>();

  uint64_t in_volume = getTensorVolume(inputTy);
  uint64_t out_volume = getTensorVolume(resultTy);

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

  TensorType inputTy = getOperand()->getType().cast<TensorType>();
  TensorType resultTy = getResult()->getType().cast<TensorType>();

  uint64_t in_volume = getTensorVolume(inputTy);
  uint64_t out_volume = getTensorVolume(resultTy);

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

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// sub_
std::map<std::string, uint64_t> SubUnderOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = getResult()->getType().cast<TensorType>();
  TensorType aType = getOperand(0)->getType().cast<TensorType>();
  Type bType = getOperand(1)->getType();

  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = getTensorVolume(aType);
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// threshold_backward
std::map<std::string, uint64_t> ThresholdBackwardOp::getStatistics() {

  std::map<std::string, uint64_t> toReturn;
  uint64_t loss_in_volume = getTensorVolume(getOperand(0)->getType().cast<TensorType>());
  uint64_t act_in_volume  = getTensorVolume(getOperand(1)->getType().cast<TensorType>()); 
  uint64_t loss_out_volume = getTensorVolume(getResult()->getType().cast<TensorType>());

  toReturn["reads"]  = loss_in_volume + act_in_volume;
  toReturn["writes"] = loss_out_volume;

  return toReturn;
}

// transpose can be zero overhead
std::map<std::string, uint64_t> TransposeOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = 0;
  toReturn["writes"] = 0;
  return toReturn;
}
// view can be zero overhead
std::map<std::string, uint64_t> ViewOp::getStatistics() {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"] = 0;
  toReturn["writes"] = 0;
  return toReturn;
}

} // namespace aten
} // namespace xilinx