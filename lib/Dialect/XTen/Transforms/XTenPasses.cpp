// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//#include "aten/Dialect/XTen/XTenDialect.h"
#include "aten/Dialect/XTen/XTenPasses.h"
//#include "aten/Dialect/XTen/XTenOps.h"

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