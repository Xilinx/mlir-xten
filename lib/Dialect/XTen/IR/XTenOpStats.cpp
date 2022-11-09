//===- XTenOpStats.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"

#include "xten/Util/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"

#include <iostream>
#include <type_traits>

#define DEBUG_TYPE "xten-op-stats"

// This file contains the StatisticsOpInterface implementations
// for ATDialect operations

using namespace mlir;
using namespace xilinx;

namespace {

template<class T>
std::map<std::string, uint64_t> getConv2dStatisticsWithType(T o, TensorType resultTy) {
    std::map<std::string, uint64_t> toReturn;

    TensorType inputTy = o.getInput().getType().template cast<TensorType>();
    TensorType weightTy = o.getWeight().getType().template cast<TensorType>();
    TensorType biasTy;
    if(o.getBias()) {
        biasTy = o.getBias().getType().template cast<TensorType>();
    }


    uint64_t ofm_volume = xilinx::xten::getTensorVolume(resultTy);
    //uint64_t ofm_depth = resultTy.getShape()[1];

    uint64_t ifm_depth = inputTy.getShape()[1];
    uint64_t kernel_height = weightTy.getShape()[2];
    uint64_t kernel_width = weightTy.getShape()[3];

    auto co = cast<arith::ConstantOp>(o.getGroups().getDefiningOp());
    auto ia = co->template getAttrOfType<IntegerAttr>("value");
    uint64_t groups = ia.getValue().getZExtValue();
    // Number of forward MACs per pixel =
    //  kernel_width * kernel_height * ifm_depth / groups
    uint64_t MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
    uint64_t total_MACs = ofm_volume * MACs_per_OFM;

    uint64_t ifm_volume = xilinx::xten::getTensorVolume(inputTy);
    uint64_t weight_volume = xilinx::xten::getTensorVolume(weightTy);
    uint64_t bias_volume;
    if(o.getBias()) {
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

template<class T>
uint64_t getConv2dOperandTransferVolumeWithType(T o, unsigned int idx, bool read, TensorType resultTy) {

  if (!read) return 0;

  double vol = xilinx::xten::getTensorVolume(o.getOperand(idx).getType());
  if (simple_conv2d_model)
    return vol;

  TensorType inputTy = o.getInput().getType().template cast<TensorType>();
  TensorType weightTy = o.getWeight().getType().template cast<TensorType>();

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

  TensorType inputTy = o.getInput().getType().template cast<TensorType>();

  if (simple_conv2d_model) {
    if (write)
      return xilinx::xten::getTensorVolume(resultTy);
    else
      return 0;
  }

  TensorType weightTy = o.getWeight().getType().template cast<TensorType>();
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

  double vol = xilinx::xten::getTensorVolume(resultTy);

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

template<>
std::map<std::string, uint64_t> getConv2dStatistics<xilinx::xten::PartialConv2dOp>(xilinx::xten::PartialConv2dOp o) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dStatisticsWithType(o, resultType);
}

template<>
std::map<std::string, uint64_t> getConv2dStatistics<xilinx::xten::PartialConv2dReLUOp>(xilinx::xten::PartialConv2dReLUOp o) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dStatisticsWithType(o, resultType);
}

template<>
std::map<std::string, uint64_t> getConv2dStatistics<xilinx::xten::PartialConv2dBatchNormReLUOp>(xilinx::xten::PartialConv2dBatchNormReLUOp o) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dStatisticsWithType(o, resultType);
}


// OperandTransferVolume
template<class T>
uint64_t getConv2dOperandTransferVolume(T o, unsigned int idx, bool read) {
    TensorType resultType = o.getResult().getType().template cast<TensorType>();
    return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

template<>
uint64_t getConv2dOperandTransferVolume<xilinx::xten::PartialConv2dOp>(xilinx::xten::PartialConv2dOp o, unsigned int idx, bool read) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

template<>
uint64_t getConv2dOperandTransferVolume<xilinx::xten::PartialConv2dReLUOp>(xilinx::xten::PartialConv2dReLUOp o, unsigned int idx, bool read) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

template<>
uint64_t getConv2dOperandTransferVolume<xilinx::xten::PartialConv2dBatchNormReLUOp>(xilinx::xten::PartialConv2dBatchNormReLUOp o, unsigned int idx, bool read) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dOperandTransferVolumeWithType(o, idx, read, resultType);
}

// ResultTransferVolume
template<class T>
uint64_t  getConv2dResultTransferVolume(T o, unsigned int idx, bool write) {
    TensorType resultType = o.getResult().getType().template cast<TensorType>();
    return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}

template<>
uint64_t getConv2dResultTransferVolume<xilinx::xten::PartialConv2dOp>(xilinx::xten::PartialConv2dOp o, unsigned int idx, bool write) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}

template<>
uint64_t getConv2dResultTransferVolume<xilinx::xten::PartialConv2dReLUOp>(xilinx::xten::PartialConv2dReLUOp o, unsigned int idx, bool write) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}

template<>
uint64_t getConv2dResultTransferVolume<xilinx::xten::PartialConv2dBatchNormReLUOp>(xilinx::xten::PartialConv2dBatchNormReLUOp o, unsigned int idx, bool write) {
    TensorType resultType = o.getResult(0).getType().template cast<TensorType>();
    return getConv2dResultTransferVolumeWithType(o, idx, write, resultType);
}


} // namespace

namespace xilinx {
namespace xten {

// acap conv2d bn relu

std::map<std::string, uint64_t> Conv2dBatchNormReLUOp::getStatistics() {
  return getConv2dStatistics<Conv2dBatchNormReLUOp>(*this);
}

uint64_t Conv2dBatchNormReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dBatchNormReLUOp>(*this, idx, read);
}

uint64_t Conv2dBatchNormReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dBatchNormReLUOp>(*this, idx, write);
}

// acap conv2d relu

std::map<std::string, uint64_t> Conv2dReLUOp::getStatistics() {
  return getConv2dStatistics<Conv2dReLUOp>(*this);
}

uint64_t Conv2dReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dReLUOp>(*this, idx, read);
}

uint64_t Conv2dReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dReLUOp>(*this, idx, write);
}

// acap conv2d

std::map<std::string, uint64_t> Conv2dOp::getStatistics() {
  return getConv2dStatistics<Conv2dOp>(*this);
}

uint64_t Conv2dOp::getOperandTransferVolume(unsigned int idx, bool read) {
  return getConv2dOperandTransferVolume<Conv2dOp>(*this, idx, read);
}

uint64_t Conv2dOp::getResultTransferVolume(unsigned int idx, bool write) {
  return getConv2dResultTransferVolume<Conv2dOp>(*this, idx, write);
}

    // TODO verify correctness of this
    std::map<std::string, uint64_t> PartialConv2dOp::getStatistics() {
        return getConv2dStatistics<PartialConv2dOp>(*this);
    }

    uint64_t PartialConv2dOp::getOperandTransferVolume(unsigned int idx, bool read) {
        return getConv2dOperandTransferVolume<PartialConv2dOp>(*this, idx, read);
    }

    uint64_t PartialConv2dOp::getResultTransferVolume(unsigned int idx, bool write) {
        return getConv2dResultTransferVolume<PartialConv2dOp>(*this, idx, write);
    }

    std::map<std::string, uint64_t> PartialConv2dReLUOp::getStatistics() {
        return getConv2dStatistics<PartialConv2dReLUOp>(*this);
    }

    uint64_t PartialConv2dReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
        return getConv2dOperandTransferVolume<PartialConv2dReLUOp>(*this, idx, read);
    }

    uint64_t PartialConv2dReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
        return getConv2dResultTransferVolume<PartialConv2dReLUOp>(*this, idx, write);
    }

    std::map<std::string, uint64_t> PartialConv2dBatchNormReLUOp::getStatistics() {
        return getConv2dStatistics<PartialConv2dBatchNormReLUOp>(*this);
    }

    uint64_t PartialConv2dBatchNormReLUOp::getOperandTransferVolume(unsigned int idx, bool read) {
        return getConv2dOperandTransferVolume<PartialConv2dBatchNormReLUOp>(*this, idx, read);
    }

    uint64_t PartialConv2dBatchNormReLUOp::getResultTransferVolume(unsigned int idx, bool write) {
        return getConv2dResultTransferVolume<PartialConv2dBatchNormReLUOp>(*this, idx, write);
    }

    std::map<std::string, uint64_t> ConcatOp::getStatistics() {
        std::map<std::string, uint64_t> toReturn;

        toReturn["ops:+"] = 0;
        toReturn["ops:MAC"] = 0;

        // NOTE assumes concat is a network communication step
        toReturn["reads"] = 0;
        toReturn["writes"] = 0;

        return toReturn;
    }

    uint64_t ConcatOp::getOperandTransferVolume(unsigned int idx, bool read) {
        // TODO
        return 0;
    }

    uint64_t ConcatOp::getResultTransferVolume(unsigned int idx, bool write) {
        // TODO
        return 0;
    }

    std::map<std::string, uint64_t> SplitOp::getStatistics() {
        std::map<std::string, uint64_t> toReturn;

        toReturn["ops:+"] = 0;
        toReturn["ops:MAC"] = 0;

        // NOTE assumes concat is a network communication step
        toReturn["reads"] = 0;
        toReturn["writes"] = 0;

        return toReturn;
    }

    uint64_t SplitOp::getOperandTransferVolume(unsigned int idx, bool read) {
        // TODO
        return 0;
    }

    uint64_t SplitOp::getResultTransferVolume(unsigned int idx, bool write) {
        // TODO
        return 0;
    }


}
}
