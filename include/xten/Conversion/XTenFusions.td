/*
Note: The generated cpp file (XtenFusions.cpp.inc) from this table-gen file has
been copied over into lib/Conversion folder in order to be able to set the
layer_name attribute on all Xten operations. It is not possible to do this here
using table-driven rewrite rules because the layer_name attribute has not been
defined for any Xten operations.

Note: If you'd like to add a new pattern in this file, you have to un-comment
this file and the CMakeLists.txt, copy the generated cpp file again into the
lib/conversion folder, and set layer_name attribute using setLayerNameAttr
function (defined in the cloned lib/conversion/AtenToXten.cpp.inc) accordingly.



//===- XTenFusions.td --------------------------------------*- tablegen -*-===//
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

// These patterns perform XTen -> XTen fusions, after the initial ATen -> XTen.
// These patterns apply on fully legalized XTen to avoid situations where an ATen op
// matches a catch-all pattern. For instance, if the two following patterns
//  (XTen_AddOp (XTen_Conv2dOp) $other)
//  (XTen_AddOp (XTen_Conv2dOp) (XTen_Conv2dOp))
// were applied in the ATenToXTen pass directly, `$other` could match an `aten.convolution`
// which has not been legalized to an `XTen_Conv2dOp` yet. If it had been legalized then
// the second pattern would match, not the first. The driver does this because it considers
// large matches before small ones, so the first pattern above is preferred to just
// `aten.convolution -> xten.conv2d` which touches only one op.
// This problem is avoided by applying all simple conversions first, then doing more fusions
// on pure XTen here.


def SelectFirstC2dInSymmetricTensorAdd : Constraint<
    CPred<"fuseFirstC2dInTensorAdd($0, $1)">,
    "fuse first parameter into tensor add, otherwise fuse second.">;

// This pattern catches the case where 2 conv2ds are inputs to an add.
// Then we callback `isLongestBranch` to find which one to fuse.
// If the callback is true (left c2d needs to be fused) then the pattern is applied.
// Otherwise another pattern is applied (benefits take care of that).
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

// This adds an average pool after the add (+ (leaky) relu)
def : Pat<(XTen_GlobalAveragePool2D (XTen_Conv2dTensorAddOp $a,$b,$c,$d,$e,$f,$g,$h)),
          (XTen_Conv2dTensorAddGlobalAveragePoolOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(XTen_GlobalAveragePool2D (XTen_Conv2dTensorAddReLUOp $a,$b,$c,$d,$e,$f,$g,$h)),
          (XTen_Conv2dTensorAddReLUGlobalAveragePoolOp $a,$b,$c,$d,$e,$f,$g,$h)>;

def : Pat<(XTen_GlobalAveragePool2D (XTen_Conv2dTensorAddLReLUOp $a,$b,$c,$d,$e,$f,$g,$alpha,$h)),
          (XTen_Conv2dTensorAddLReLUGlobalAveragePoolOp $a,$b,$c,$d,$e,$f,$g,$alpha,$h)>;

*/