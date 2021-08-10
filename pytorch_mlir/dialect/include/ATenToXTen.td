// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "npcomp/Dialect/ATen/IR/ATenOps.td"
include "XTenOps.td"

def : Pat<(aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
          (XTen_Conv2dOp  $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_TOp (aten_TOp $input)), (XTen_NoOp $input)>;

def : Pat<(aten_ReluOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_ReluUnderOp (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i)),
          (XTen_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_ReluOp
            (aten_NativeBatchNormOp
              (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (XTen_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_ReluUnderOp
            (aten_NativeBatchNormOp
              (XTen_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (XTen_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_AddOp $a, (ConstantOp:$b $ab), (ConstantOp:$c $ac)),
          (XTen_AddConstantOp $a, $b)>;

def : Pat<(aten_MmOp $a, $b),
          (XTen_MMOp $a, $b)>;

def : Pat<(aten_MulOp $a, $b),
          (XTen_MulOp $a, $b)>;

def : Pat<(aten_AddOp $a, $b, $c),
          (XTen_AddOp $a, $b)>;
