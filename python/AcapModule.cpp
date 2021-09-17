// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

// LLVM and MLIR headers must come first
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"

#include "xten/Transform/Passes.h"
#include "xten/Conversion/Passes.h"

#include "xten/Transform/ATenOpReport.h"
#include "xten/Transform/LivenessReport.h"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace llvm;

namespace llvm {
extern bool DebugFlag;
}

namespace xten {
namespace {

void InitXTenModuleBindings(pybind11::module m)
{

  m.def("_register_all_passes", []() {
    xilinx::xten::registerTransformPasses();
    xilinx::xten::registerConversionPasses();
  }, "register all passes");

}

} // namespace

void InitXTenBindings(pybind11::module m) { InitXTenModuleBindings(m); }

}  // namespace xten

PYBIND11_MODULE(_xten, m) { xten::InitXTenBindings(m); }
