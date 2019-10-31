// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "ATenToStd.h"
#include "ATenDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"

using namespace xilinx;

namespace {
// import patterns
#include "ATenToStd.cpp.inc"
} // namespace

namespace mlir {
void populateATenToStdPatterns(MLIRContext *context,
                              OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
}
}