#ifdef MLIR_ATEN_TO_STD_TD
#else
#define MLIR_ATEN_TO_STD_TD

#ifdef STANDARD_OPS
#else
include "mlir/Dialect/StandardOps/Ops.td"
#endif // STANDARD_OPS

#ifdef ATEN_OPS
#else
include "aten.td"
#endif

//def : Pat<(aten_ThresholdOp $x, (aten_ConstantOp:$a1), (aten_ConstantOp:$a2)),(tb_ReLUOp $x)>;

#endif
