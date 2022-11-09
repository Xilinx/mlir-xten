//===- XTenOps.td ------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_OPS
#define XTEN_OPS

include "xten/Dialect/XTen/XTenBase.td"

include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class XTen_Op<string mnemonic, list<Trait> traits = []>
    : Op<XTen_Dialect, mnemonic, traits> {
}

def XTen_AddConstantOp: XTen_Op<"add_constant", []>,
                  Results<(outs AnyTorchTensorType:$output)> {
  let arguments = (
    ins AnyTorchTensorType:$src,
        AnyType:$c
  );

  let summary = "add one operator";
  let description = [{
    add one operator
  }];
}

def XTen_AddOp: XTen_Op<"add", []>,
               Results<(outs AnyTorchTensorType:$output)> {
  let arguments = (
    ins AnyTorchTensorType:$input0,
        AnyTorchTensorType:$input1
  );

  let summary = "add operator";
  let description = [{
    add operator
  }];
}

def XTen_MMOp: XTen_Op<"mm", [Pure]>,
              Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$x,
        AnyTorchTensorType:$y
  );

  let summary = "matrix multiply operator";
  let description = [{
    matrix multiply operator
  }];
}


def XTen_MulOp: XTen_Op<"mul", []>,
               Results<(outs AnyTorchTensorType:$output)> {
  let arguments = (
    ins AnyTorchTensorType:$input0,
        AnyTorchTensorType:$input1
  );

  let summary = "mul operator";
  let description = [{
    mul operator
  }];
}

def XTen_SoftmaxOp: XTen_Op<"softmax", []>,
               Results<(outs AnyTorchTensorType:$output)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        Torch_IntType:$dim,
        Torch_BoolType:$half_to_float
  );

  let summary = "Softmax operator";
  let description = [{
    Computes the softmax across the specified dimension.
  }];
}

def XTen_GlobalAveragePool2D: XTen_Op<"globalaveragepool2d", [Pure]>,
               Results<(outs AnyTorchTensorType:$output)> {
  let arguments = (
    ins AnyTorchTensorType:$input
  );

  let summary = "GlobalAveragePool2D operator";
  let description = [{
    Computes the average pool along the spatial dimensions. Rank of the input
    and output are equivalent. Spatial dimensions are collapsed to one.
  }];
}

def XTen_NoOp: XTen_Op<"noop", []>,
                Results<(outs AnyType)> {
  let arguments = (
    ins AnyType:$x
  );

  let summary = "noop returns its input";
  let description = [{
    noop returns its input or a copy of its input
  }];
}

def XTen_Conv2dOp: XTen_Op<"conv2d", [Pure]>,
                                Results<(outs AnyTorchTensorType:$result)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups
  );

  let summary = "Convolution operator";
  let description = [{
    Convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

// TODO what happens when we have both?
def XTen_PartialConv2dOp: XTen_Op<"partialconv2d", [Pure]>{
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchOptionalTensorType:$PartialIn,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups
  );

  let results = (
      outs AnyTorchTensorType:$output,
           AnyTorchOptionalTensorType:$forward
  );

  let summary = "Partial convolution operator";
  let description = [{
    Partial convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}


def XTen_Conv2dReLUOp: XTen_Op<"conv2d_relu", [Pure]>,
                                   Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups
  );

  let summary = "Convolution ReLU operator";
  let description = [{
    Fused Convolution ReLU operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_PartialConv2dReLUOp: XTen_Op<"partialconv2d_relu",
                                [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchOptionalTensorType:$PartialIn,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups
  );

  let results = (
      outs AnyTorchTensorType:$output,
           AnyTorchOptionalTensorType:$forward
  );

  let summary = "Partial convolution ReLU operator";
  let description = [{
    Quantized convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dBatchNormReLUOp: XTen_Op<"conv2d_bn_relu", [Pure]>,
                                            Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        AnyTorchTensorType:$bn_weight,
        AnyTorchTensorType:$bn_bias,
        AnyTorchTensorType:$running_mean,
        AnyTorchTensorType:$running_var,
        Torch_BoolType:$training,
        Torch_FloatType:$momentum,
        Torch_FloatType:$eps

  );

  let summary = "Convolution BatchNorm ReLU operator";
  let description = [{
    Fused Convolution BatchNorm ReLU operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_PartialConv2dBatchNormReLUOp: XTen_Op<"partialconv2d_bn_relu",
                                        [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchOptionalTensorType:$PartialIn,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        AnyTorchTensorType:$bn_weight,
        AnyTorchTensorType:$bn_bias,
        AnyTorchTensorType:$running_mean,
        AnyTorchTensorType:$running_var,
        Torch_BoolType:$training,
        Torch_FloatType:$momentum,
        Torch_FloatType:$eps
  );

  let results = (
      outs AnyTorchTensorType:$output,
           AnyTorchOptionalTensorType:$forward
  );

  let summary = "Partial Convolution BatchNorm ReLU operator";
  let description = [{
    Fused Convolution BatchNorm ReLU operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_ConcatOp: XTen_Op<"concat", [Pure]>,
                                Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins Variadic<AnyTorchTensorType>:$inputs,
        XTen_AnyScalar:$dim
  );

  let summary = "Concat operator";
  let description = [{
    Concat operator
  }];
  let extraClassDeclaration = [{ // TODO might remove these declarations
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

// TODO Proper verifier for this operation?
def XTen_SplitOp: XTen_Op<"split", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        XTen_AnyScalar:$dim
  );

  let results = (
      outs Variadic<AnyTorchTensorType>:$outputs
  );

  let summary = "split operator";
  let description = [{
    split operator
  }];
  let extraClassDeclaration = [{ // TODO might remove these declarations
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dLReLUOp: XTen_Op<"conv2d_lrelu", [Pure]>,
                                   Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_FloatType:$alpha
  );

  let summary = "Convolution with Leaky ReLU operator";
  let description = [{
    Fused Convolution followed by Leaky ReLU activation operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dLReLUPadOp: XTen_Op<"conv2d_lrelu_pad", [Pure]>,
                                   Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_FloatType:$alpha,
        Torch_ListType:$pad_padding,
        AnyTorchScalarType:$pad_value
  );

  let summary = "Convolution with Leaky ReLU plus Pad operator";
  let description = [{
    Fused Convolution followed by Leaky ReLU activation and Pad operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}


def XTen_Conv2dReLUPadOp: XTen_Op<"conv2d_relu_pad", [Pure]>,
                                   Results<(outs AnyTorchTensorType)> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_ListType:$pad_padding,
        AnyTorchScalarType:$pad_value
  );

  let summary = "Convolution with ReLU plus Pad operator";
  let description = [{
    Fused Convolution followed by Leaky ReLU activation and Pad operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}


def XTen_Conv2dReLUMaxPoolOp: XTen_Op<"conv2d_relu_maxpool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_ListType:$mp_kernel_size,
        Torch_ListType:$mp_stride,
        Torch_ListType:$mp_padding,
        Torch_ListType:$mp_dilation,
        Torch_BoolType:$mp_ceil_mode
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with ReLU plus MaxPool operator";
  let description = [{
    Fused Convolution followed by ReLU activation followed by compatible MaxPool operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dReLUPadMaxPoolOp: XTen_Op<"conv2d_relu_pad_maxpool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_ListType:$pad_padding,
        AnyTorchScalarType:$pad_value,
        Torch_ListType:$mp_kernel_size,
        Torch_ListType:$mp_stride,
        Torch_ListType:$mp_padding,
        Torch_ListType:$mp_dilation,
        Torch_BoolType:$mp_ceil_mode
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with ReLU plus Pad and MaxPool operator";
  let description = [{
    Fused Convolution followed by ReLU activation followed by compatible Pad and MaxPool operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}


def XTen_Conv2dLReLUMaxPoolOp: XTen_Op<"conv2d_lrelu_maxpool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_FloatType:$alpha,
        Torch_ListType:$mp_kernel_size,
        Torch_ListType:$mp_stride,
        Torch_ListType:$mp_padding,
        Torch_ListType:$mp_dilation,
        Torch_BoolType:$mp_ceil_mode
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with Leaky ReLU plus MaxPool operator";
  let description = [{
    Fused Convolution followed by Leaky ReLU activation followed by compatible MaxPool operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dLReLUPadMaxPoolOp: XTen_Op<"conv2d_lrelu_pad_maxpool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,
        Torch_FloatType:$alpha,
        Torch_ListType:$pad_padding,
        AnyTorchScalarType:$pad_value,
        Torch_ListType:$mp_kernel_size,
        Torch_ListType:$mp_stride,
        Torch_ListType:$mp_padding,
        Torch_ListType:$mp_dilation,
        Torch_BoolType:$mp_ceil_mode
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with Leaky ReLU plus Pad and MaxPool operator";
  let description = [{
    Fused Convolution followed by Leaky ReLU activation followed by compatible Pad and MaxPool operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool write);
	}];
}

def XTen_Conv2dTensorAddOp: XTen_Op<"conv2d_tensoradd", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor";
  let description = [{
    Convolution with linear activation, then element-wise add with another tensor.
  }];
}

def XTen_Conv2dTensorAddReLUOp: XTen_Op<"conv2d_tensoradd_relu", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor, then ReLU";
  let description = [{
    Convolution with linear activation, then element-wise add with another tensor, then ReLU.
  }];
}

def XTen_Conv2dTensorAddLReLUOp: XTen_Op<"conv2d_tensoradd_lrelu", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        Torch_FloatType:$alpha, /// for lrelu

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor, then leaky ReLU";
  let description = [{
    Convolution with linear activation, then element-wise add with another tensor, then leaky ReLU.
  }];
}

def XTen_Conv2dTensorAddGlobalAveragePoolOp: XTen_Op<"conv2d_tensoradd_globalaveragepool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor, then global average pool.";
  let description = [{
    Convolution with linear activation, then element-wise add with another
    tensor, then average pool.
  }];
}

def XTen_Conv2dTensorAddReLUGlobalAveragePoolOp: XTen_Op<"conv2d_tensoradd_relu_globalaveragepool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor, then ReLU, then global average pool.";
  let description = [{
    Convolution with linear activation, then element-wise add with another
    tensor, then ReLU, then average pool.
  }];
}

def XTen_Conv2dTensorAddLReLUGlobalAveragePoolOp: XTen_Op<"conv2d_tensoradd_lrelu_globalaveragepool", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias,
        Torch_ListType:$stride,
        Torch_ListType:$padding,
        Torch_ListType:$dilation,
        Torch_IntType:$groups,

        Torch_FloatType:$alpha, /// for lrelu

        AnyTorchTensorType:$add_ifm
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Convolution with linear activation, then element-wise add with another tensor, then leaky ReLU, then global average pool.";
  let description = [{
    Convolution with linear activation, then element-wise add with another
    tensor, then leaky ReLU, then average pool.
  }];
}

def XTen_LinearOp: XTen_Op<"linear", [Pure]> {
  let arguments = (
    ins AnyTorchTensorType:$input,
        AnyTorchTensorType:$weight,
        AnyTorchOptionalTensorType:$bias
  );

  let results = (
      outs AnyTorchTensorType:$output
  );

  let summary = "Applies a linear function Y = XA^T+B to the input function";
  let description = [{
    Applies a linear function to the input Y = XA^T+B. Represents a fully connected layer.
  }];
}


#endif // #ifndef XTEN_OPS
