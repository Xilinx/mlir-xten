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

def : Pat<(Torch_AtenConv2dOp $a,$b,$c,$d,$e,$f,$g),
          (XTen_Conv2dOp  $a,$b,$c,$d,$e,$f,$g)>;

// def : Pat<(aten_TOp (aten_TOp $input)), (XTen_NoOp $input)>;

def : Pat<(Torch_AtenReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g)>;

def : Pat<(Torch_AtenRelu_Op (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g)>;

// def : Pat<(aten_LeakyReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),$h),
//           (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

// def : Pat<(aten_LeakyReluOp 
//             (Torch_AtenBatchNormOp
//               (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g),
//               $a1,$a2,$a3,$a4,$a5,$a6,$a7),$j),
//           (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h)>;

// def : Pat<(Torch_AtenMaxPool2dOp
//             (XTen_Conv2dLReLUOp $a,$b,$c,$d,$e,$f,$g,$h),$mp2,$mp3,$mp4,$mp5,$mp6),
//           (XTen_Conv2dLReLUMaxPoolOp $a,$b,$c,$mp3,$e,$f,$g,$h)>;

// def : Pat<(Torch_AtenReluOp
//             (Torch_AtenBatchNormOp
//               (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
//               $a1,$a2,$a3,$a4,$a5,$a6,$a7,$a8
//             )
//           ),
//           (XTen_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

// def : Pat<(Torch_AtenRelu_Op
//             (Torch_AtenBatchNormOp
//               (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
//               $a1,$a2,$a3,$a4,$a5,$a6,$a7,$a8
//             )
//           ),
//           (XTen_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(Torch_AtenAdd_TensorOp $a, (ConstantOp:$b $ab), (ConstantOp:$c $ac)),
          (XTen_AddConstantOp $a, $b)>;

def : Pat<(Torch_AtenMmOp $a, $b),
          (XTen_MMOp $a, $b)>;

def : Pat<(Torch_AtenMulTensorOp $a, $b),
          (XTen_MulOp $a, $b)>;

def : Pat<(Torch_AtenAddTensorOp $a, $b, $c),
          (XTen_AddOp $a, $b)>;
