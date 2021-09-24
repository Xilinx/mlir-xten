#ifndef XTEN_PASSES_H
#define XTEN_PASSES_H

#include "mlir/Pass/Pass.h"

#include "xten/Dialect/XTen/XTenDataflow.h"
#include "xten/Dialect/XTen/XTenNamePass.h"

namespace xilinx {
namespace xten {

void registerXTenPasses();

}
}

#endif // XTEN_PASSES_H
