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

// This conversion operates XTen -> XTen, after the initial ATen -> XTen.


def SelectFirstC2dInSymmetricTensorAdd : Constraint<
    CPred<"fuseFirstC2dInTensorAdd($0, $1)">,
    "fuse first parameter into tensor add, otherwise fuse second.">;

// This pattern catches the case where 2 conv2ds are inputs to an add.
// Then we callback `fuseFirstC2dInTensorAdd` to find which one to fuse.
// If the callback is true (left c2d needs to be fused) then the pattern is applied.
// Otherwise another pattern is applied (benefits care about that).
def : Pat<(XTen_AddOp (XTen_Conv2dOp:$c2d0 $a,$b,$c,$d,$e,$f,$g), (XTen_Conv2dOp:$c2d1 $_,$_,$_,$_,$_,$_,$_)),
          (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g, $c2d1),
          [(SelectFirstC2dInSymmetricTensorAdd $c2d0, $c2d1)],
          (addBenefit 3)>;

def : Pat<(XTen_AddOp (XTen_Conv2dOp:$c2d0 $_,$_,$_,$_,$_,$_,$_), (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g, $c2d0),
          [],
          (addBenefit 2)>;

def : Pat<(XTen_AddOp $h, (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(XTen_AddOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g), $h),
          (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)>;

// these add an activation after the tensoradd

def : Pat<(Torch_AtenReluOp (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)),
          (XTen_Conv2dTensorAddReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(Torch_AtenLeakyReluOp (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h), $alpha),
          (XTen_Conv2dTensorAddLReLUOp $a,$b,$c,$d,$e,$f,$g,$alpha,$h)>;
