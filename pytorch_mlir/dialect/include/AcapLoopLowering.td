// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef ACAP_LOOP_LOWERING
#define ACAP_LOOP_LOWERING

include "mlir/IR/OpBase.td"
include "include/ATen.td"
include "include/AIR.td"

// def : Pat<(aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
//           (air_Conv2dOp  $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;

#endif
