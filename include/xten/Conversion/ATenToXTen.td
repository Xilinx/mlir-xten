/*
Note: The generated cpp file (AtenToXten.cpp.inc) from this table-gen file has
been copied over into lib/Conversion folder in order to be able to set the
layer_name attribute on all Xten operations. It is not possible to do this here
using table-driven rewrite rules because the layer_name attribute has not been
defined for any Aten operations.

Note: If you'd like to add a new pattern in this file, you have to un-comment
this file and the CMakeLists.txt, copy the generated cpp file again into the
lib/conversion folder, and set layer_name attribute using setLayerNameAttr
function (defined in the cloned lib/conversion/AtenToXten.cpp.inc) accordingly.


//===- ATenToXTen.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"

include "torch-mlir/Dialect/Torch/IR/TorchOps.td"

include "xten/Dialect/XTen/XTenOps.td"

// note: this pass is complemented by XTenFusions.td

// Helper Check Functions
def CheckTrue : Constraint<
    CPred<"llvm::dyn_cast<Torch::ConstantBoolOp>($0.getDefiningOp()).value()">,
    "check to see if it is torch.constant.bool true">;

def CheckFalse : Constraint<
    CPred<"!llvm::dyn_cast<Torch::ConstantBoolOp>($0.getDefiningOp()).value()">,
    "check to see if it is torch.constant.bool false">;

def CheckEmptyList : Constraint<
    CPred<"$0.getDefiningOp<Torch::PrimListConstructOp>().getNumOperands() == 0">,
    "check to see if it is torch.list is empty">;

// Conversions

def : Pat<(Torch_AtenConvolutionOp $input,$weight,$bias,$stride,$padding,$dilation,$transposed,$output_padding,$groups),
          (XTen_Conv2dOp  $input,$weight,$bias,$stride,$padding,$dilation,$groups),
          [(CheckFalse $transposed), (CheckEmptyList $output_padding)]>;

def : Pat<(Torch_AtenReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g)>;

def : Pat<(Torch_AtenLeakyReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),$h),
          (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenConstantPadNdOp (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h), $pad,$val),
          (XTen_Conv2dLReLUPadOp $a,$b,$c,$d,$e,$f,$g,$h,$pad,$val)>;

def : Pat<(Torch_AtenConstantPadNdOp (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g), $pad,$val),
          (XTen_Conv2dReLUPadOp $a,$b,$c,$d,$e,$f,$g,$pad,$val)>;
// def : Pat<(aten_LeakyReluOp 
//             (Torch_AtenBatchNormOp
//               (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),
//               $a1,$a2,$a3,$a4,$a5,$a6,$a7),$j),
//           (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenMaxPool2dOp
            (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h),$mp2,$mp3,$mp4,$mp5,$mp6), 
          (XTen_Conv2dLReLUMaxPoolOp $a,$b,$c,$d,$e,$f,$g,$h,$mp2,$mp3,$mp4,$mp5,$mp6)>;

def : Pat<(Torch_AtenMaxPool2dOp
            (XTen_Conv2dLReLUPadOp $a,$b,$c,$d,$e,$f,$g,$h,$pad,$val),$mp2,$mp3,$mp4,$mp5,$mp6), 
          (XTen_Conv2dLReLUPadMaxPoolOp $a,$b,$c,$d,$e,$f,$g,$h,$pad,$val,$mp2,$mp3,$mp4,$mp5,$mp6)>;
// def : Pat<(Torch_AtenMaxPool2dOp(
//             Torch_AtenConstantPadNdOp 
//               (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h),$pad, $val),$mp2,$mp3,$mp4,$mp5,$mp6), 
//           (XTen_Conv2dLReLUMaxPoolOp $a,$b,$c,$d,$e,$f,$g,$h,$mp2,$mp3,$mp4,$mp5,$mp6)>;


def : Pat<(Torch_AtenMaxPool2dOp
            (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g),$mp2,$mp3,$mp4,$mp5,$mp6),
          (XTen_Conv2dReLUMaxPoolOp $a,$b,$c,$d,$e,$f,$g,$mp2,$mp3,$mp4,$mp5,$mp6)>;

def : Pat<(Torch_AtenMaxPool2dOp
            (XTen_Conv2dReLUPadOp $a,$b,$c,$d,$e,$f,$g,$pad,$val),$mp2,$mp3,$mp4,$mp5,$mp6),
          (XTen_Conv2dReLUPadMaxPoolOp $a,$b,$c,$d,$e,$f,$g,$pad,$val,$mp2,$mp3,$mp4,$mp5,$mp6)>;

def : Pat<(Torch_AtenReluOp
            (Torch_AtenBatchNormOp
              (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7,$a8
            )
          ),
          (XTen_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(Torch_AtenAdd_TensorOp $a, (ConstantOp:$b $ab), (ConstantOp:$c $ac)),
          (XTen_AddConstantOp $a, $b)>;

def : Pat<(Torch_AtenMmOp $a, $b),
          (XTen_MMOp $a, $b)>;

def : Pat<(Torch_AtenMulTensorOp $a, $b),
          (XTen_MulOp $a, $b)>;

// TODO: verify that $c == 1.
def : Pat<(Torch_AtenAddTensorOp $a, $b, $c),
          (XTen_AddOp $a, $b)>;

def : Pat<(Torch_Aten_SoftmaxOp $a, $b, $c),
          (XTen_SoftmaxOp $a, $b, $c)>;

// Convert to global average pool based on constraints
def CheckReduceMean : Constraint<
    CPred<"isReduceMeanGlobalAveragePool2D($0, $1)">,
    "check if attributes are valid for global average pool">;

def : Pat<(Torch_AtenMeanDimOp $input, $dims, $keepdim, $reduce),
          (XTen_GlobalAveragePool2D $input),
          [(CheckReduceMean $dims, $keepdim)]>;

def CheckAvgPool2D : Constraint<
    CPred<"isAvgPoolGlobalAveragePool2D($0, $1, $2, $3)">,
    "check if attributes of AvgPool2D are valid for global average pool">;

def : Pat<(Torch_AtenAvgPool2dOp $input, $kernel_size, $stride, $padding, (Torch_ConstantBoolOp:$ceil_mode $cm), (Torch_ConstantBoolOp:$count_padding $cp), (Torch_ConstantNoneOp)),
          (XTen_GlobalAveragePool2D $input),
          [(CheckAvgPool2D $input, $kernel_size, $stride, $padding), (CheckFalse $ceil_mode), (CheckTrue $count_padding)]>;

def CheckAdaptiveAvgPool : Constraint<
    CPred<"isAdaptiveAvgPoolGlobalAveragePool2D($0)">,
    "check if attributes are valid for global average pool">;

def : Pat<(Torch_AtenAdaptiveAvgPool2dOp $input, $outsizes),
          (XTen_GlobalAveragePool2D $input),
          [(CheckAdaptiveAvgPool $outsizes)]>;

def CheckLinearForXten : Constraint<
    CPred<"checkLinearForXten($0, $1, $2)">,
    "Only specific ranks for tensors are supported.\n"
    "Check the ranks of the inputs:\n"
    "    y = A*B^T + C\n"
    "Where rank(A) == 2, rank(B) == 2 and rank(C) == 1.\n"
    "Note, C must be defined. It cannot be set to none">;

def : Pat<(Torch_AtenLinearOp $input, $weights, $bias),
          (XTen_LinearOp  $input, $weights, $bias),
          [(CheckLinearForXten $input, $weights, $bias)]>;

*/
