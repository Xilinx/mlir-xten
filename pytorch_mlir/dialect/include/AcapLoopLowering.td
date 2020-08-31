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
