// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef ATEN_OPS
#else
#define ATEN_OPS
include "include/ATen.td"
#endif

#ifdef AIR_OPS
#else
#define AIR_OPS
include "include/AIR.td"
#endif

def : Pat<(aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
          (air_Conv2dOp  $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

def : Pat<(aten_TransposeOp (aten_TransposeOp $input)), (air_NoOp $input)>;

def : Pat<(aten_ReLUOp (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

def : Pat<(aten_ReLUUnderOp (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

def : Pat<(aten_ReLUOp
            (aten_NativeBatchNormOp
              (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_ReLUUnderOp
            (aten_NativeBatchNormOp
              (air_Conv2dOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_AddOp $a, (aten_ConstantOp:$b), (aten_ConstantOp:$c)),
          (air_AddOneOp $a, $b)>;

def : Pat<(aten_AddOp $a, F32Tensor:$b, $c),
          (air_AddOneOp $a, $b)>;

def : Pat<(aten_MMOp $a, $b),
          (air_MMOp $a, $b)>;

def : Pat<(aten_MulOp $a, $b),
          (air_MulOp $a, $b)>;
