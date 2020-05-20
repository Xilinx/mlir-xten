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

#ifdef ACAP_OPS
#else
#define ACAP_OPS
include "include/acap.td"
#endif

def : Pat<(aten_ConvolutionOp $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l),
          (aten_AcapConv2dOp  $a,$b,$c,$d,$e,$f,$g,$h,$i,$j,$k,$l)>;
