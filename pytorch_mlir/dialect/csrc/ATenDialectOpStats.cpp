#include "ATenDialect.h"

namespace {

uint64_t getTensorVolume(TensorType ty) {
  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(MemRefType ty) {
  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(Type ty) {
  if (auto t = ty.cast<TensorType>()) {
    return getTensorVolume(t);
  }
  else if (auto t = ty.cast<MemRefType>()) {
    return getTensorVolume(t);
  }
  else {
    return 0;
  }
}

} // namespace

namespace xilinx {
namespace aten {

// add
std::map<std::string, uint64_t> AddOp::updateStatistics() {

  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult()->getType();
  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  uint64_t a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return toReturn;
}

// add_
std::map<std::string, uint64_t> AddUnderOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult()->getType();
  uint64_t ofm_volume = getTensorVolume(resultTy);

  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  uint64_t a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return toReturn;
}

// addmm
std::map<std::string, uint64_t> AddmmOp::updateStatistics() {

  std::map<std::string, uint64_t> toReturn;
  // For linear, we need the number of output neurons and the number of input neurons
  // Then the number of forward MACs is input * output
  // And the number of adds is output if there is bias

  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t num_output_neurons = tensorResultTy.getShape()[1];
  uint64_t ofm_volume = getTensorVolume(tensorResultTy);

  // Use the weight tensor to find the number of input neurons
  Type wType = getOperand(2)->getType();
  TensorType wTy = wType.cast<TensorType>();
  uint64_t num_input_neurons = wTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  uint64_t weight_volume = getTensorVolume(wTy);

  Type ifmType = getOperand(1)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  uint64_t ifm_volume = getTensorVolume(txTy);

  toReturn["MAC"] = total_MACs;
  toReturn["+"] = ofm_volume;   // Should be gated on whether there is bias at all
  toReturn["activation_in"] = ifm_volume;
  toReturn["activation_out"] = ofm_volume;
  toReturn["parameters_in"] = weight_volume + num_output_neurons;

  return toReturn;
}

// batch_norm
std::map<std::string, uint64_t>  BatchNormOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;

  // There are 2x as many parameters are there are planes ...
  uint64_t ifm_depth = tensorResultTy.getShape()[1];
  toReturn["parameters_in"] = ifm_depth * 2;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares

  toReturn["+"] = op_volume;   // Add up for mean
  toReturn["*"] = op_volume;   // Square for variance
  toReturn["+"] += op_volume;  // Add up squares for variance

  toReturn["*"] += ifm_depth;   // Calc channel means
  toReturn["-"] += ifm_depth;   // Calc channel vars
  toReturn["*"] += ifm_depth;   // Calc channel vars

  toReturn["sqrt"] = ifm_depth;  // Convert to SD
  toReturn["/"] += ifm_depth;    // Get the reciprocal

  toReturn["+"] += op_volume;   // Subtract mean off each pixel
  toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["+"] += op_volume;   // Bias
  toReturn["*"] += op_volume;   // Scale

  return toReturn;
}

// _convolution
std::map<std::string, uint64_t> ConvolutionOp::updateStatistics() {

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
  toReturn["+"] = ofm_volume;

  toReturn["MAC"] = total_MACs;
  toReturn["activation_in"] = ifm_volume;
  toReturn["activation_out"] = ofm_volume;
  toReturn["parameters_in"] = weight_volume + bias_volume;

  return toReturn;
}

// max_pool2d
std::map<std::string, uint64_t> MaxPool2dOp::updateStatistics() {

  auto unpack = [](auto &op, auto &v) -> void {
                  auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                  for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
                };

  std::map<std::string, uint64_t>  toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_out"] = ofm_volume;

  Type ifmType = getOperand(0)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  uint64_t ifm_volume = getTensorVolume(txTy);
  toReturn["activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel;
  mlir::Value *k = getOperand(1);
  unpack(k, kernel);

  uint64_t aperture = kernel[0] * kernel[1];
  toReturn[">"] = ofm_volume * aperture;

  return toReturn;
}

// max_pool2d_with_indices
std::map<std::string, uint64_t> MaxPool2dWithIndicesOp::updateStatistics() {

  auto unpack = [](auto &op, auto &v) -> void {
                  auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                  for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
                };

  std::map<std::string, uint64_t>  toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_out"] = ofm_volume;

  Type ifmType = getOperand(0)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  uint64_t ifm_volume = getTensorVolume(txTy);
  toReturn["activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel;
  mlir::Value *k = getOperand(1);
  unpack(k, kernel);

  uint64_t aperture = kernel[0] * kernel[1];
  toReturn[">"] = ofm_volume * aperture;

  return toReturn;
}

// mul
std::map<std::string, uint64_t> MulOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult()->getType();
  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  uint64_t a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return toReturn;
}

// mul_
std::map<std::string, uint64_t> MulUnderOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;

  Type resultTy = getResult()->getType();
  uint64_t ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  uint64_t a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  uint64_t b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return toReturn;
}

// native_batch_norm
std::map<std::string, uint64_t> NativeBatchNormOp::updateStatistics() {
  std::map<std::string, uint64_t>  toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;

  // There are 2x as many parameters are there are planes ...
  uint64_t ifm_depth = tensorResultTy.getShape()[1];
  toReturn["parameters_in"] = ifm_depth * 2;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares

  toReturn["+"] = op_volume;   // Add up for mean
  toReturn["*"] = op_volume;   // Square for variance
  toReturn["+"] += op_volume;  // Add up squares for variance

  toReturn["*"] += ifm_depth;   // Calc channel means
  toReturn["-"] += ifm_depth;   // Calc channel vars
  toReturn["*"] += ifm_depth;   // Calc channel vars

  toReturn["sqrt"] = ifm_depth;  // Convert to SD
  toReturn["/"] += ifm_depth;    // Get the reciprocal

  toReturn["+"] += op_volume;   // Subtract mean off each pixel
  toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["+"] += op_volume;   // Bias
  toReturn["*"] += op_volume;   // Scale

  return toReturn;
}

// relu
std::map<std::string, uint64_t> ReLUOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;
  toReturn[">"] = op_volume;

  return toReturn;
}

// relu_
std::map<std::string, uint64_t> ReLUUnderOp::updateStatistics() {
  std::map<std::string, uint64_t> toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  uint64_t op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;
  toReturn[">"] = op_volume;

  return toReturn;
}

} // namespace aten
} // namespace xilinx