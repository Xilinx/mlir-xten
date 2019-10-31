// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
// ATenToTB.h
#ifndef MLIR_ATEN_TO_STD_H
#define MLIR_ATEN_TO_STD_H

#include "mlir/Transforms/DialectConversion.h"

namespace xilinx {
namespace aten {
    class ATenDialect;
}
}

namespace mlir {

void populateATenToStdPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_ATEN_TO_STD_H