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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class XTen_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<XTen_Dialect, mnemonic, traits> {
}

def XTen_AddConstantOp: XTen_Op<"add_constant", []>,
                  Results<(outs AnyTensor:$output)> {
  let arguments = (
    ins AnyTensor:$src,
        AnyType:$c
  );

  let summary = "add one operator";
  let description = [{
    add one operator
  }];
}

def XTen_AddOp: XTen_Op<"add", []>,
               Results<(outs AnyTensor:$output)> {
  let arguments = (
    ins AnyTensor:$input0,
        AnyTensor:$input1
  );

  let summary = "add operator";
  let description = [{
    add operator
  }];
}

def XTen_MMOp: XTen_Op<"mm", [NoSideEffect, SameOperandsAndResultElementType]>,
              Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyTensor:$y
  );

  let summary = "matrix multiply operator";
  let description = [{
    matrix multiply operator
  }];
}


def XTen_MulOp: XTen_Op<"mul", []>,
               Results<(outs AnyTensor:$output)> {
  let arguments = (
    ins AnyTensor:$input0,
        AnyTensor:$input1
  );

  let summary = "mul operator";
  let description = [{
    mul operator
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

def XTen_Conv2dOp: XTen_Op<"conv2d", [NoSideEffect]>,
                                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups
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
def XTen_PartialConv2dOp: XTen_Op<"partialconv2d", [NoSideEffect]>{
  let arguments = (
    ins AnyTensor:$input,
        Optional<AnyTensor>:$PartialIn,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups
  );

  let results = (
      outs AnyTensor:$output,
           Optional<AnyTensor>:$forward
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


def XTen_Conv2dReLUOp: XTen_Op<"conv2d_relu", [NoSideEffect]>,
                                   Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups
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

def XTen_PartialConv2dReLUOp: XTen_Op<"partialconv2d_relu", [NoSideEffect]> {
  let arguments = (
    ins AnyTensor:$input,
        Optional<AnyTensor>:$PartialIn,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups
  );

  let results = (
      outs AnyTensor:$output,
           Optional<AnyTensor>:$forward
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

def XTen_Conv2dBatchNormReLUOp: XTen_Op<"conv2d_bn_relu", [NoSideEffect]>,
                                            Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups,
        AnyTensor:$bn_weight,
        AnyTensor:$bn_bias,
        AnyTensor:$running_mean,
        AnyTensor:$running_var,
        AnyType:$training,
        XTen_AnyScalar:$momentum,
        XTen_AnyScalar:$eps

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

def XTen_PartialConv2dBatchNormReLUOp: XTen_Op<"partialconv2d_bn_relu", [NoSideEffect]> {
  let arguments = (
    ins AnyTensor:$input,
        Optional<AnyTensor>:$PartialIn,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups,
        AnyTensor:$bn_weight,
        AnyTensor:$bn_bias,
        AnyTensor:$running_mean,
        AnyTensor:$running_var,
        AnyType:$training,
        XTen_AnyScalar:$momentum,
        XTen_AnyScalar:$eps
  );

  let results = (
      outs AnyTensor:$output,
           Optional<AnyTensor>:$forward
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

def XTen_ConcatOp: XTen_Op<"concat", [NoSideEffect]>,
                                Results<(outs AnyTensor)> {
  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
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
def XTen_SplitOp: XTen_Op<"split", [NoSideEffect]> {
  let arguments = (
    ins AnyTensor:$input,
        XTen_AnyScalar:$dim
  );

  let results = (
      outs Variadic<AnyTensor>:$outputs
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

def XTen_Conv2dLReLUOp: XTen_Op<"conv2d_lrelu", [NoSideEffect]>,
                                   Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups,
        AnyType:$alpha
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


def XTen_Conv2dLReLUMaxPoolOp: XTen_Op<"conv2d_lrelu_maxpool", [NoSideEffect]> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        XTen_AnyOptionalTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyType:$transposed,
        AnyType:$output_padding,
        XTen_AnyScalar:$groups,
        AnyType:$alpha
  );

  let results = (
      outs AnyTensor:$output,
           AnyTensor:$indices
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


#endif // #ifndef XTEN_OPS
