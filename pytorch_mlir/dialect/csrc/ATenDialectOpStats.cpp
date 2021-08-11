#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"

#include "Util.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>
#include <type_traits>

#define DEBUG_TYPE "aten-op-stats"

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;
using namespace xilinx;

namespace {

    void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
        if (auto co = op.getDefiningOp<NPCOMP::aten::ConstantOp>()) {
            DenseElementsAttr a = co->template getAttrOfType<DenseElementsAttr>("value");
            for (auto i : a.getIntValues())
                v.push_back(i.getSExtValue());
        }
        else if (auto co = op.getDefiningOp<NPCOMP::Basicpy::BuildListOp>()) {
            for (auto o : op.getDefiningOp()->getOperands())
                v.push_back(o.template getDefiningOp<ConstantIntOp>().getValue());
        }
    }

template<class T>
std::map<std::string, uint64_t> getConv2dStatisticsWithType(T o, TensorType resultTy) {
    std::map<std::string, uint64_t> toReturn;

    TensorType inputTy = o.input().getType().template cast<TensorType>();
    TensorType weightTy = o.weight().getType().template cast<TensorType>();
    TensorType biasTy;
    if(!o.bias().template getDefiningOp<NPCOMP::Basicpy::SingletonOp>()) {
        biasTy = o.bias().getType().template cast<TensorType>();
    }


    uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
    //uint64_t ofm_depth = resultTy.getShape()[1];

    uint64_t ifm_depth = inputTy.getShape()[1];
    uint64_t kernel_height = weightTy.getShape()[2];
    uint64_t kernel_width = weightTy.getShape()[3];

    auto co = cast<ConstantOp>(o.groups().getDefiningOp());
    auto ia = co->template getAttrOfType<IntegerAttr>("value");
    uint64_t groups = ia.getValue().getZExtValue();
    // Number of forward MACs per pixel =
    //  kernel_width * kernel_height * ifm_depth / groups
    uint64_t MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
    uint64_t total_MACs = ofm_volume * MACs_per_OFM;

    uint64_t ifm_volume = xilinx::aten::getTensorVolume(inputTy);
    uint64_t weight_volume = xilinx::aten::getTensorVolume(weightTy);
    uint64_t bias_volume;
    if(!o.bias().template getDefiningOp<NPCOMP::Basicpy::SingletonOp>()) {
        bias_volume = xilinx::aten::getTensorVolume(biasTy);
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

template<class T>
uint64_t getConv2dOperandTransferVolumeWithType(T o, unsigned int idx, bool read, TensorType resultTy) {

  if (!read) return 0;

  double vol = xilinx::aten::getTensorVolume(o.getOperand(idx).getType());
  if (simple_conv2d_model)
    return vol;

  TensorType inputTy = o.input().getType().template cast<TensorType>();
  TensorType weightTy = o.weight().getType().template cast<TensorType>();

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
uint64_t getConv2dResultTransferVolumeWithType(T o, unsigned int idx, bool write, TensorType resultTy) {

  TensorType inputTy = o.input().getType().template cast<TensorType>();

  if (simple_conv2d_model) {
    if (write)
      return xilinx::aten::getTensorVolume(resultTy);
    else
      return 0;
  }

  TensorType weightTy = o.weight().getType().template cast<TensorType>();
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

  double vol = xilinx::aten::getTensorVolume(resultTy);

  if (write) {
    LLVM_DEBUG(llvm::outs() << "write_output_overhead:" << write_output_overhead << "\n");
    return vol * write_output_overhead;
  } else {
    LLVM_DEBUG(llvm::outs() << "read_output_cost:" << read_output_cost << "\n");
    return vol * read_output_cost;
  }
}

// TODO can this indirection be cleaned?


// Conv2dStatistics
template<class T>
std::map<std::string, uint64_t> getConv2dStatistics(T o) {
    TensorType resultType = o.getResult().getType().template cast<TensorType>();
    return getConv2dStatisticsWithType(o, resultType);
}

// OperandTransferVolume
template<class T>
uint64_t getConv2dOperandTransferVolume(T o, unsigned int idx, bool read) {
    TensorType resultType = o.getResult().getType().template cast<TensorType>();
    return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

// ResultTransferVolume
template<class T>
uint64_t  getConv2dResultTransferVolume(T o, unsigned int idx, bool write) {
    TensorType resultType = o.getResult().getType().template cast<TensorType>();
    return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}

} // namespace

#if 1

namespace xilinx {
namespace aten {

using namespace mlir::NPCOMP::aten;
template<class OpT>
std::map<std::string, uint64_t> getStatistics(OpT op) {
  return std::map<std::string, uint64_t>();
}


// add
template<>
std::map<std::string, uint64_t> getStatistics(AddOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// add_
template<>
std::map<std::string, uint64_t> getStatistics(AddUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// addmm
template<>
std::map<std::string, uint64_t> getStatistics(AddmmOp op) {

  std::map<std::string, uint64_t> toReturn;

  // For linear, we need the number of output neurons and the number of input neurons
  // Then the number of forward MACs is input * output
  // And the number of adds is output if there is bias

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType biasTy = op.getOperand(0).getType().cast<TensorType>();
  TensorType inputTy = op.getOperand(1).getType().cast<TensorType>();
  TensorType weightTy = op.getOperand(2).getType().cast<TensorType>();

  uint64_t num_output_neurons = resultTy.getShape()[1];
  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  uint64_t weight_volume = xilinx::aten::getTensorVolume(weightTy);

  uint64_t ifm_volume = xilinx::aten::getTensorVolume(inputTy);

  toReturn["ops:MAC"] = total_MACs;
  toReturn["ops:+"] = ofm_volume;   // Should be gated on whether there is bias at all
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:0:parameters_in:bias"] = xilinx::aten::getTensorVolume(biasTy);
  toReturn["operand:2:parameters_in:weight"] = weight_volume;

  toReturn["reads"] = ifm_volume + weight_volume + num_output_neurons;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

    // as_strided can be zero overhead
    template<>
    std::map<std::string, uint64_t> getStatistics(AsStridedOp op) {
        std::map<std::string, uint64_t> toReturn;
        toReturn["reads"] = 0;
        toReturn["writes"] = 0;
        toReturn["operand:0:activation_in"] = 0;
        toReturn["result:0:activation_out"] = 0;
        return toReturn;
    }

// batch_norm
template<>
std::map<std::string, uint64_t> getStatistics(BatchNormOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = xilinx::aten::getTensorVolume(resultTy);
  uint64_t weight_volume = xilinx::aten::getTensorVolume(op.getOperand(1).getType());
  uint64_t bias_volume = xilinx::aten::getTensorVolume(op.getOperand(2).getType());
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
template<>
std::map<std::string, uint64_t> getStatistics(ConvolutionOp op) {
  return getConv2dStatistics<ConvolutionOp>(op);
}

uint64_t getOperandTransferVolume(ConvolutionOp op, unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<ConvolutionOp>(op, idx, read);
}

uint64_t getResultTransferVolume(ConvolutionOp op, unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<ConvolutionOp>(op, idx, write);
}

// _convolution_backward
template<>
std::map<std::string, uint64_t> getStatistics(ConvolutionBackwardOp op) {

  std::map<std::string, uint64_t> toReturn;
  TensorType dx_out_resultTy = op.getResult(0).getType().cast<TensorType>();
  uint64_t dx_out_volume = xilinx::aten::getTensorVolume(dx_out_resultTy);

  TensorType weightTy = op.getOperand(2).getType().cast<TensorType>();
  uint64_t weight_volume = xilinx::aten::getTensorVolume(weightTy);
  uint64_t loss_in_depth = weightTy.getShape()[0];
  uint64_t kernel_width = weightTy.getShape()[2];
  uint64_t kernel_height = weightTy.getShape()[3];

  uint64_t groups = 1; // todo: get this in the same way as the forward path
  uint64_t MACs_per_loss = (loss_in_depth/groups) * kernel_height * kernel_width;

  uint64_t total_MACs = dx_out_volume * MACs_per_loss;

  TensorType ifmTy = op.getOperand(1).getType().cast<TensorType>();
  uint64_t ifm_volume = xilinx::aten::getTensorVolume(ifmTy);
  auto ifm_shape = ifmTy.getShape();

  uint64_t ifm_bwh = ifm_shape[0]*ifm_shape[2]*ifm_shape[3];  // Batch * height * width: the depth is in the weight shape already
  total_MACs += ifm_bwh * weight_volume;

  TensorType dx_inTy = op.getOperand(0).getType().cast<TensorType>();
  uint64_t dx_in_volume = xilinx::aten::getTensorVolume(dx_inTy);
  toReturn["ops:+"] = dx_in_volume;

  // Reads: Conv_backward reads 3 tensors: the loss in, the activation in and the transposed weights
  toReturn["reads"] = dx_in_volume + ifm_volume + weight_volume;

  // Writes: Conv_backward writes 3 tensors: the loss out, gradients for the weights, and gradients for the biases
  TensorType biasTy = op.getResult(2).getType().cast<TensorType>();
  uint64_t bias_volume = xilinx::aten::getTensorVolume(biasTy);
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
template<>
std::map<std::string, uint64_t> getStatistics(DivOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;

  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// div_
template<>
std::map<std::string, uint64_t> getStatistics(DivUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  toReturn["ops:/"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;


  return toReturn;
}

// expand can be zero overhead
template<>
std::map<std::string, uint64_t> getStatistics(ExpandOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// flatten can be zero overhead
template<>
std::map<std::string, uint64_t> getStatistics(FlattenOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// hardtanh
template<>
std::map<std::string, uint64_t> getStatistics(HardtanhOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = op.getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = op.getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::aten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// hardtanh_
template<>
std::map<std::string, uint64_t> getStatistics(HardtanhUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = op.getOperand(0).getType().cast<TensorType>();
  TensorType resultTy = op.getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::aten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// max_pool2d
template<>
std::map<std::string, uint64_t> getStatistics(MaxPool2dOp op) {

  std::map<std::string, uint64_t> toReturn;

  // TensorType resultTy = op.getResult().getType().cast<TensorType>();
  // TensorType inputType = op.getOperand(0).getType().cast<TensorType>();

  // uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  // toReturn["result:0:activation_out"] = ofm_volume;

  // uint64_t ifm_volume = xilinx::aten::getTensorVolume(inputType);
  // toReturn["input:0:activation_in"] = ifm_volume;

  // // To find the number of compares, we need the filter extent

  // std::vector<uint64_t> kernel_size = unpackListConstant(op.getOperand(1));

  // uint64_t aperture = kernel_size[0] * kernel_size[1];
  // toReturn["ops:>"] = ofm_volume * (aperture-1);

  // toReturn["reads"] = ifm_volume;
  // toReturn["writes"] = ofm_volume;

  return toReturn;
}

// max_pool2d_with_indices
template<>
std::map<std::string, uint64_t> getStatistics(MaxPool2dWithIndicesOp op) {

    std::map<std::string, uint64_t> toReturn;

    uint64_t ofm_volume = xilinx::aten::getTensorVolume(op.getResult(0).getType().cast<TensorType>());
    uint64_t indices_volume = xilinx::aten::getTensorVolume(op.getResult(1).getType().cast<TensorType>());

    toReturn["writes"] = ofm_volume + indices_volume;
    toReturn["result:0:activation_out"] = ofm_volume;
    toReturn["result:1:indices_out"] = indices_volume;

    uint64_t ifm_volume = xilinx::aten::getTensorVolume(op.getOperand(0).getType().cast<TensorType>());
    toReturn["reads"] = ifm_volume;
    toReturn["operand:0:activation_in"] = ifm_volume;

    // To find the number of compares, we need the filter extent
    std::vector<int64_t> kernel_size;
    unpack_int_list(op.getOperand(1), kernel_size);

    uint64_t aperture = kernel_size[0] * kernel_size[1];
    toReturn["ops:>"] = ofm_volume * (aperture-1);

  return toReturn;
}

// max_pool2d_with_indicies_backward
template<>
std::map<std::string, uint64_t> getStatistics(MaxPool2dWithIndicesBackwardOp op) {

  std::map<std::string, uint64_t> toReturn;

  Type resultTy = op.getResult().getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  uint64_t loss_out_volume = xilinx::aten::getTensorVolume(tensorResultTy);
  toReturn["writes"] = loss_out_volume;

  uint64_t loss_in_volume = xilinx::aten::getTensorVolume(op.getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume  = xilinx::aten::getTensorVolume(op.getOperand(1).getType().cast<TensorType>()); // TODO: Why is this needed?
  uint64_t indices_volume  = xilinx::aten::getTensorVolume(op.getOperand(7).getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + act_in_volume + indices_volume;
  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = act_in_volume;
  toReturn["operand:3:activation_in"] = indices_volume;
  toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// mean
template<>
std::map<std::string, uint64_t> getStatistics(MeanOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand().getType().cast<TensorType>();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  toReturn["ops:+"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);

  toReturn["operand:0:activation_in"] = a_volume;

  toReturn["reads"] = a_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mm
template<>
std::map<std::string, uint64_t> getStatistics(MmOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  TensorType weightTy = op.getOperand(1).getType().cast<TensorType>();
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  toReturn["ops:MAC"] = total_MACs;

  uint64_t loss_in_volume = xilinx::aten::getTensorVolume(op.getOperand(0).getType().cast<TensorType>());
  uint64_t weight_volume = xilinx::aten::getTensorVolume(op.getOperand(1).getType().cast<TensorType>());
  toReturn["reads"] = loss_in_volume + weight_volume;
  toReturn["writes"] = ofm_volume;

  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = weight_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  return toReturn;
}

// mul
template<>
std::map<std::string, uint64_t> getStatistics(MulOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// mul_
template<>
std::map<std::string, uint64_t> getStatistics(MulUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);
  toReturn["ops:*"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// native_batch_norm
template<>
std::map<std::string, uint64_t> getStatistics(NativeBatchNormOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult(0).getType().cast<TensorType>();
  uint64_t op_volume = xilinx::aten::getTensorVolume(resultTy);
  uint64_t weight_volume = xilinx::aten::getTensorVolume(op.getOperand(1).getType());
  uint64_t bias_volume = xilinx::aten::getTensorVolume(op.getOperand(2).getType());
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
template<>
std::map<std::string, uint64_t> getStatistics(NativeBatchNormBackwardOp op) {

  std::map<std::string, uint64_t> toReturn;

  ShapedType inputTy = op.getOperand(0).getType().cast<ShapedType>();
  uint64_t input_volume = xilinx::aten::getTensorVolume(inputTy);
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
    auto v = xilinx::aten::getTensorVolume(op.getOperand(i).getType());
    toReturn["operand:"+std::to_string(i)+":activation_in"] = v;
    reads += v;
  }

  uint64_t writes = 0;
  for (int i=0; i<3; i++) {
    auto v = xilinx::aten::getTensorVolume(op.getResult(i).getType());
    toReturn["result:"+std::to_string(i)+":grad"] = v;
    writes += v;
  }

  toReturn["reads"] = reads;
  toReturn["writes"] = writes;

  return toReturn;
}

// relu
template<>
std::map<std::string, uint64_t> getStatistics(ReluOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = op.getOperand().getType().cast<TensorType>();
  TensorType resultTy = op.getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::aten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// relu_
template<>
std::map<std::string, uint64_t> getStatistics(ReluUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = op.getOperand().getType().cast<TensorType>();
  TensorType resultTy = op.getResult().getType().cast<TensorType>();

  uint64_t in_volume = xilinx::aten::getTensorVolume(inputTy);
  uint64_t out_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"]  = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

// sub
template<>
std::map<std::string, uint64_t> getStatistics(SubOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;

}

// sub_
template<>
std::map<std::string, uint64_t> getStatistics(SubUnderOp op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().cast<TensorType>();
  TensorType aType = op.getOperand(0).getType().cast<TensorType>();
  Type bType = op.getOperand(1).getType();

  uint64_t ofm_volume = xilinx::aten::getTensorVolume(resultTy);

  toReturn["ops:-"] = ofm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;

  // Find the size of the A and B operands
  uint64_t a_volume = xilinx::aten::getTensorVolume(aType);
  uint64_t b_volume = xilinx::aten::getTensorVolume(bType);

  toReturn["operand:0:activation_in"] = a_volume;
  toReturn["operand:1:activation_in"] = b_volume;

  toReturn["reads"] = a_volume + b_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// sum
template<>
std::map<std::string, uint64_t> getStatistics(SumOp op) {

  std::map<std::string, uint64_t> toReturn;
  TensorType ty = op.getOperand(0).getType().cast<TensorType>();
  uint64_t volume = xilinx::aten::getTensorVolume(ty);

  toReturn["ops:+"] = volume;

  toReturn["operand:0:activation_in"] = volume;
  toReturn["result:0:activation_out"] = volume;

  toReturn["reads"] = volume;
  toReturn["writes"] = volume;

  return toReturn;
}

// threshold_backward
template<>
std::map<std::string, uint64_t> getStatistics(ThresholdBackwardOp op) {

  std::map<std::string, uint64_t> toReturn;
  uint64_t loss_in_volume = xilinx::aten::getTensorVolume(op.getOperand(0).getType().cast<TensorType>());
  uint64_t act_in_volume  = xilinx::aten::getTensorVolume(op.getOperand(1).getType().cast<TensorType>());
  uint64_t loss_out_volume = xilinx::aten::getTensorVolume(op.getResult().getType().cast<TensorType>());

  toReturn["reads"]  = toReturn["operand:0:activation_in"] = loss_in_volume + act_in_volume;
  toReturn["writes"] = toReturn["result:0:grad:dx"] = loss_out_volume;

  return toReturn;
}

// transpose can be zero overhead
// template<>
// std::map<std::string, uint64_t> getStatistics(TransposeOp op) {
//   std::map<std::string, uint64_t> toReturn;
//   toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
//   toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
//   return toReturn;
// }

// unsqueeze can be zero overhead
template<>
std::map<std::string, uint64_t> getStatistics(UnsqueezeOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

// view can be zero overhead
template<>
std::map<std::string, uint64_t> getStatistics(aten::ViewOp op) {
  std::map<std::string, uint64_t> toReturn;
  toReturn["reads"]  = toReturn["operand:0:activation_in"] = 0;
  toReturn["writes"] = toReturn["result:0:activation_out"] = 0;
  return toReturn;
}

std::map<std::string, uint64_t> getATenOpStats(Operation *op)
{

#define GET_STATS(T) \
  if (isa<T>(op)) return getStatistics<T>( cast<T>(op) );

  GET_STATS(AddOp)
  GET_STATS(AddUnderOp)
  GET_STATS(AddmmOp)
  GET_STATS(AsStridedOp)
  GET_STATS(BatchNormOp)
  GET_STATS(ConvolutionOp)
  GET_STATS(ConvolutionBackwardOp)
  GET_STATS(DivOp)
  GET_STATS(DivUnderOp)
  GET_STATS(ExpandOp)
  GET_STATS(FlattenOp)
  GET_STATS(HardtanhOp)
  GET_STATS(HardtanhUnderOp)
  GET_STATS(MaxPool2dOp)
  GET_STATS(MaxPool2dWithIndicesOp)
  GET_STATS(MaxPool2dWithIndicesBackwardOp)
  GET_STATS(MeanOp)
  GET_STATS(MmOp)
  GET_STATS(MulOp)
  GET_STATS(MulUnderOp)
  GET_STATS(NativeBatchNormOp)
  GET_STATS(NativeBatchNormBackwardOp)
  GET_STATS(ReluOp)
  GET_STATS(ReluUnderOp)
  GET_STATS(SubOp)
  GET_STATS(SubUnderOp)
  GET_STATS(SumOp)
  GET_STATS(ThresholdBackwardOp)
//  GET_STATS(TransposeOp)
  GET_STATS(UnsqueezeOp)
  GET_STATS(aten::ViewOp)

  return std::map<std::string, uint64_t>();
}

} // namespace aten
} // namespace xilinx
#endif
