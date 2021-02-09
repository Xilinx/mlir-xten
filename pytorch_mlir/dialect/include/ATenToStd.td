// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifndef ATEN_TO_STD_TD
#define ATEN_TO_STD_TD

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "ATen.td"

//def : Pat<(aten_ThresholdOp $x, (aten_ConstantOp:$a1), (aten_ConstantOp:$a2)),(tb_ReLUOp $x)>;

#endif
