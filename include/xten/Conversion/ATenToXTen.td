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
include "mlir/Dialect/StandardOps/IR/Ops.td"

include "torch-mlir/Dialect/Torch/IR/TorchOps.td"

include "xten/Dialect/XTen/XTenOps.td"

// Normalize names, so that we only have to write one pattern.
// Each operator comes in two variants, eg torch.aten.relu and torch.aten.relu_ (with underscore)
// PyTorch operators have an in-place variant and a by-return variant,
// which have the same MLIR representation because of SSA, but different
// operators.
def : Pat<(Torch_AtenRelu_Op $a),
          (Torch_AtenReluOp $a)>;
def : Pat<(Torch_AtenLeakyRelu_Op $a, $b),
          (Torch_AtenLeakyReluOp $a, $b)>;
def : Pat<(Torch_AtenAdd_TensorOp $a, $b, $c),
          (Torch_AtenAddTensorOp $a, $b, $c)>;

// Conversions

def : Pat<(Torch_AtenConv2dOp $a,$b,$c,$d,$e,$f,$g),
          (XTen_Conv2dOp  $a,$b,$c,$d,$e,$f,$g)>;

def : Pat<(Torch_AtenReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g)>;

def : Pat<(Torch_AtenLeakyReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),$h),
          (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenConstantPadNdOp (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h), $pad,$val),
          (XTen_Conv2dLReLUPadOp $a,$b,$c,$d,$e,$f,$g,$h,$pad,$val)>;
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

def : Pat<(Torch_AtenAddTensorOp $a, $b, $c),
          (XTen_AddOp $a, $b)>;

def : Pat<(XTen_AddOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g), $h),
          (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenReluOp (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)),
          (XTen_Conv2dTensorAddReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenLeakyReluOp (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h), $alpha),
          (XTen_Conv2dTensorAddLReLUOp $a,$b,$c,$d,$e,$f,$g,$alpha,$h)>;
