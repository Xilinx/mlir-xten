// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir-c/Core.h"

#include "ATenDialect.h"
#include "ForwardPathReport.h"

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace xilinx {
namespace aten {

void init(void) {
  mlir::registerDialect<xilinx::aten::ATenDialect>();
}

mlir_type_t make_list_type(mlir_type_t elemType)
{
  mlir::Type eTy = mlir::Type::getFromOpaquePointer(elemType);
  return mlir_type_t{ ATenListType::get(eTy).getAsOpaquePointer() };
}

} // namespace aten
} // namespace xilinx

namespace {
  
mlir::OwningModuleRef LoadModule(mlir::MLIRContext &context, std::string mlir) {

  mlir::OwningModuleRef module;

  std::unique_ptr<llvm::MemoryBuffer> membuf = llvm::MemoryBuffer::getMemBuffer(mlir);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(membuf), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Error can't parse mlir module\n";
    return nullptr;
  }
  if (failed(mlir::verify(*module))) {
    llvm::errs() << "Error verifying MLIR module\n";
    return nullptr;
  }
  if (!module)
    return nullptr;
  return module;
}

}

PYBIND11_MODULE(pybind_aten, m) {
  m.doc() = "Python bindings for Xilinx ATen MLIR Dialect";
  m.def("version", []() { return "0.1"; });
  m.def("make_list_type", &xilinx::aten::make_list_type, "make ATenListType");
  m.def("init", &xilinx::aten::init, "register dialect");
  m.def("fwd_path_report_pass", [](std::string mlir) -> std::string {
    mlir::MLIRContext context;
    auto module = LoadModule(context, mlir);
    mlir::PassManager pm(module->getContext());

    // our pass
    std::string report;
    pm.addPass(xilinx::reports::createForwardPathReportPass(true/*useJSON*/, report));

    if (failed(pm.run(*module))) {
      llvm::errs() << "ForwardPathReportPass failed";
      return "<error>";
    }
    return report;
  }, "run ForwardPathReportPass");
}