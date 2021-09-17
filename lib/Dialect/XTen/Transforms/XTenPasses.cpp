// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenPasses.h"
//#include "xten/Dialect/XTen/XTenOps.h"

using namespace mlir;
using namespace xilinx::xten;


namespace xilinx {
namespace xten {

void registerXTenPasses() {
#define GEN_PASS_REGISTRATION
  //registerXTenToAffinePass();
  registerXTenDataflowPass();
  registerXTenNamePass();
}

}
}