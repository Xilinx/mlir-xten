// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef AIR_OPS
#else
#define AIR_OPS
include "include/AIR.td"
#endif
