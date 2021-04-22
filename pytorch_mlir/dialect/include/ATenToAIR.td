// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "npcomp/Dialect/ATen/IR/ATenOps.td"
include "AIR.td"

def : Pat<(aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
          (air_Conv2dOp  $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_TOp (aten_TOp $input)), (air_NoOp $input)>;

def : Pat<(aten_ReluOp (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_ReluUnderOp (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i)>;

def : Pat<(aten_ReluOp
            (aten_NativeBatchNormOp
              (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_ReluUnderOp
            (aten_NativeBatchNormOp
              (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_AddOp $a, (ConstantOp:$b $ab), (ConstantOp:$c $ac)),
          (air_AddConstantOp $a, $b)>;

def : Pat<(aten_MmOp $a, $b),
          (air_MMOp $a, $b)>;

def : Pat<(aten_MulOp $a, $b),
          (air_MulOp $a, $b)>;

def : Pat<(aten_AddOp $a, $b, $c),
          (air_AddOp $a, $b)>;
