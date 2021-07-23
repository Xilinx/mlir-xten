include_directories(${PROJECT_SOURCE_DIR}/dialect/include)
include_directories(${NPCOMP_SOURCE_DIR}/include)

add_mlir_dialect(AIR air)
set(LLVM_TARGET_DEFINITIONS AIR.td)
mlir_tablegen(AIREnums.h.inc -gen-enum-decls)
mlir_tablegen(AIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIREnumsIncGen)

add_mlir_dialect(AIRRtOps airrt)
set(LLVM_TARGET_DEFINITIONS AIRRtOps.td)
add_public_tablegen_target(MLIRAIRRtIncGen)

add_mlir_dialect(XTenOps xten)
set(LLVM_TARGET_DEFINITIONS XTenOps.td)
add_public_tablegen_target(MLIRXTenIncGen)

set(LLVM_TARGET_DEFINITIONS AIROpBase.td)
mlir_tablegen(AIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIROpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIROpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRAIROpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToAIR.td)
mlir_tablegen(ATenToAIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToAIRIncGen)

#set(LLVM_TARGET_DEFINITIONS ONNXToAIR.td)
#mlir_tablegen(ONNXToAIR.cpp.inc -gen-rewriters)
#add_public_tablegen_target(MLIRONNXToAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AffineToAIR.td)
mlir_tablegen(AffineToAIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAffineToAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AcapLoopLowering.td)
mlir_tablegen(AcapLoopLowering.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAcapLoopLoweringIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToStd.td)
mlir_tablegen(ATenToStd.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToStdIncGen)

set(LLVM_TARGET_DEFINITIONS ATenPasses.td)
mlir_tablegen(ATenPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRATenPassIncGen)

add_mlir_doc(ATenPasses ConversionPasses ./ -gen-pass-doc)
add_mlir_doc(AIRPasses AIRPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS AIRToLinalg.td)
mlir_tablegen(AIRToLinalg.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAIRToLinalgIncGen)

set(LLVM_TARGET_DEFINITIONS AIRPasses.td)
mlir_tablegen(AIRPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRAIRPassIncGen)

set(LLVM_TARGET_DEFINITIONS AIRRtPasses.td)
mlir_tablegen(AIRRtPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRAIRRtPassIncGen)