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

def : Pat<(aten_TransposeOp (aten_TransposeOp $input)), (air_NoOp $input)>;

def : Pat<(aten_ReLUOp (aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

def : Pat<(aten_ReLUUnderOp (aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)),
          (air_Conv2dReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

def : Pat<(aten_ReLUOp
            (aten_NativeBatchNormOp
              (aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;

def : Pat<(aten_ReLUUnderOp
            (aten_NativeBatchNormOp
              (aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
              $a1,$a2,$a3,$a4,$a5,$a6,$a7
            )
          ),
          (air_Conv2dBatchNormReLUOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l,$a1,$a2,$a3,$a4,$a5,$a6,$a7)>;
