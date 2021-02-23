// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "ATenToStd.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace xilinx;

namespace {
// import patterns
#include "ATenToStd.cpp.inc"
} // namespace

namespace mlir {
void populateATenToStdPatterns(MLIRContext *context,
                              OwningRewritePatternList &patterns) {
  populateWithGenerated(context, patterns);
}
}
