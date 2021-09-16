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

#include "aten/Transform/Passes.h"
#include "aten/Conversion/Passes.h"

#include "aten/Transform/ATenOpReport.h"
#include "aten/Transform/LivenessReport.h"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace llvm;

namespace llvm {
extern bool DebugFlag;
}

namespace torch_acap {
namespace {

// mlir::OwningModuleRef LoadModule(mlir::MLIRContext &context, std::string mlir) {

//   mlir::OwningModuleRef module;

//   // load the dialects we expect as input
//   context.loadDialect<mlir::NPCOMP::aten::ATenDialect>();
//   context.loadDialect<mlir::StandardOpsDialect>();

//   std::unique_ptr<llvm::MemoryBuffer> membuf = llvm::MemoryBuffer::getMemBuffer(mlir);

//   llvm::SourceMgr sourceMgr;
//   sourceMgr.AddNewSourceBuffer(std::move(membuf), llvm::SMLoc());
//   module = mlir::parseSourceFile(sourceMgr, &context);

//   if (!module) {
//     llvm::errs() << "Error can't parse mlir module\n";
//     return nullptr;
//   }
//   if (failed(mlir::verify(*module))) {
//     llvm::errs() << "Error verifying MLIR module\n";
//     return nullptr;
//   }
//   if (!module)
//     return nullptr;
//   return module;
// }

void InitAcapModuleBindings(pybind11::module m)
{

  m.def("_register_all_passes", []() {
    xilinx::aten::registerTransformPasses();
    xilinx::aten::registerConversionPasses();
  }, "register all passes");

  // m.def("_op_report", [](std::string mlir) -> std::string {
  //   mlir::MLIRContext context;

  //   auto module = LoadModule(context, mlir);
  //   mlir::PassManager pm(module->getContext());

  //   // our pass
  //   std::string report;
  //   pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
  //   pm.addPass(xilinx::aten::createATenOpReportPass(report));

  //   if (failed(pm.run(*module))) {
  //     llvm::errs() << "ATenOpReportPass failed";
  //     return "<error>";
  //   }
  //   return report;
  // }, "run ATenOpReportPass");

  // m.def("_liveness_report", [](std::string mlir) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   mlir::PassManager pm(module->getContext());

  //   pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
  //   if (failed(pm.run(*module))) {
  //     llvm::errs() << "ATen generate liveness report failed";
  //     return "<error>";
  //   }

  //   auto mOp = module.get();
  //   auto liveness = xilinx::aten::LivenessReport(mOp);
  //   std::string report = liveness.emitJSONReport();
  //   return report;
  // }, "generate liveness report");

  // m.def("aten_to_air", [](std::string mlir) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   PassManager pm0(module->getContext());
  //   pm0.addPass(xilinx::aten::createATenToXTenPass());
  //   pm0.addPass(mlir::createCSEPass());

  //   if (failed(pm0.run(*module))) {
  //     llvm::errs() << "fusion pass failed ";
  //     return "";
  //   }

  //   // dump MLIR to string and return
  //   std::string s;
  //   llvm::raw_string_ostream ss(s);
  //   module->print(ss);
  //   return ss.str();
  // }, "ATen dialect to AIR dialect pass");
  
  // m.def("lower_to_affine", [](std::string mlir) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   PassManager pm0(module->getContext());
  //   pm0.addPass(xilinx::aten::createAcapLoopLoweringPass());

  //   if (failed(pm0.run(*module))) {
  //     llvm::errs() << "aten to affine conversion failed ";
  //     return "";
  //   }

  //   // dump MLIR to string and return
  //   std::string s;
  //   llvm::raw_string_ostream ss(s);
  //   module->print(ss);
  //   return ss.str();
  // }, "lower aten to affine + acap dialect");

  // m.def("lower_to_std", [](std::string mlir) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   PassManager pm0(module->getContext());
  //   pm0.addPass(xilinx::aten::createATenToXTenPass());
  //   pm0.addPass(mlir::createCSEPass());
  //   pm0.addPass(mlir::createCSEPass());
  //   pm0.addPass(xilinx::aten::createAcapLoopLoweringPass());
  //   pm0.addPass(mlir::createCSEPass());
  //   pm0.addPass(xilinx::aten::createATenLoweringPass());
  //   pm0.addPass(xilinx::aten::createReturnEliminationPass());

  //   if (failed(pm0.run(*module))) {
  //     llvm::errs() << "aten to loops conversion failed ";
  //     return "";
  //   }

  //   PassManager pm1(module->getContext());
  //   pm1.addPass(mlir::createLowerAffinePass());
  //   pm1.addPass(mlir::createLowerToCFGPass());
  //   pm1.addPass(mlir::createCSEPass());

  //   if (failed(pm1.run(*module))) {
  //     llvm::errs() << "loops to std conversion failed ";
  //     return "";
  //   }

  //   // dump MLIR to string and return
  //   std::string s;
  //   llvm::raw_string_ostream ss(s);
  //   module->print(ss);
  //   return ss.str();
  // }, "lower aten to std dialect");

  // m.def("_simple_alloc_pass", [](std::string mlir, std::string model_json) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   mlir::PassManager pm(module->getContext());

  //   pm.addPass(mlir::createCSEPass());
  //   pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
  //   pm.addPass(xilinx::aten::createATenSimpleAllocatePass(model_json));
  //   if (failed(pm.run(*module))) {
  //     llvm::errs() << "ATen simple alloc pass failed";
  //     return "<error>";
  //   }

  //   // dump MLIR to string and return
  //   std::string s;
  //   llvm::raw_string_ostream ss(s);
  //   module->print(ss);
  //   return ss.str();
  // }, "run SimpleAlloc pass");

  // m.def("_model", [](std::string mlir, std::string model_json) -> std::string {
  //   mlir::MLIRContext context;
  //   auto module = LoadModule(context, mlir);

  //   mlir::PassManager pm(module->getContext());

  //   pm.addPass(xilinx::aten::createATenToXTenPass());

  //   pm.addPass(mlir::createCSEPass());
  //   pm.addPass(mlir::createCSEPass());
  //   pm.addPass(mlir::NPCOMP::aten::createATenLayerNamePass());
  //   pm.addPass(xilinx::aten::createATenSimpleAllocatePass(model_json));
  //   if (failed(pm.run(*module))) {
  //     llvm::errs() << "ATen model failed";
  //     return "<error>";
  //   }

  //   std::stringstream traceStream;
  //   //ATenCommandProcessor proc(traceStream);

  //   //proc.run(module.get(), model_json);
  //   return traceStream.str();

  // }, "model MLIR execution");

  // m.def("affine_opt_tile_sizes", [](std::vector<uint64_t> sizes) -> void {
  //   xilinx::aten::AffineLoopOptTileSizes = sizes;
  // }, "affine loop opt pass tile sizes");

  // m.def("affine_opt_copy_depths", [](std::vector<uint64_t> depths) -> void {
  //   xilinx::aten::AffineLoopOptCopyDepths = depths;
  // }, "affine loop opt pass copy depths");

  // m.def("affine_opt_copy_fast_space", [](uint64_t space) -> void {
  //   xilinx::aten::AffineLoopOptFastSpace = space;
  // }, "affine loop opt pass fast memory space");

  // m.def("affine_opt_copy_slow_space", [](uint64_t space) -> void {
  //   xilinx::aten::AffineLoopOptSlowSpace = space;
  // }, "affine loop opt pass slow memory space");

  // m.def("conv2d_loop_order", [](std::vector<uint64_t> order) -> void {
  //   xilinx::aten::Conv2dLoopOrder = order;
  // }, "force conv2d loop order");

  // m.def("conv2d_copy_depth", [](std::vector<uint64_t> depth) -> void {
  //   xilinx::aten::Conv2dCopyDepth = depth;
  // }, "force conv2d copy depth");

  // m.def("conv2d_tile_sizes", [](std::vector<uint64_t> sizes) -> void {
  //   xilinx::aten::Conv2dTileSizes = sizes;
  // }, "force conv2d tile sizes");

  // m.def("set_debug", [](bool b, std::string &type) -> void {
  //   setCurrentDebugType(type.c_str());
  //   llvm::DebugFlag = b;
  // }, "enable/disable debug messages");
}

} // namespace

void InitAcapBindings(pybind11::module m) { InitAcapModuleBindings(m); }

}  // namespace torch_acap

PYBIND11_MODULE(_acap, m) { torch_acap::InitAcapBindings(m); }
