// (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

#ifndef XTEN_C_DIALECTS_H
#define XTEN_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(XTenNN, xten_nn);

#ifdef __cplusplus
}
#endif

#endif // XTEN_C_DIALECTS_H
