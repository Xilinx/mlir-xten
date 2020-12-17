include_directories(${PROJECT_SOURCE_DIR}/dialect)

add_mlir_dialect(ATen aten)
set(LLVM_TARGET_DEFINITIONS ATen.td)
mlir_tablegen(ATenEnums.h.inc -gen-enum-decls)
mlir_tablegen(ATenEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRATenEnumsIncGen)

add_mlir_dialect(AIR air)
set(LLVM_TARGET_DEFINITIONS AIR.td)
mlir_tablegen(AIREnums.h.inc -gen-enum-decls)
mlir_tablegen(AIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToAIR.td)
mlir_tablegen(ATenToAIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AffineToAIR.td)
mlir_tablegen(AffineToAIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAffineToAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AcapLoopLowering.td)
mlir_tablegen(AcapLoopLowering.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAcapLoopLoweringIncGen)

set(LLVM_TARGET_DEFINITIONS ATenOpInterface.td)
mlir_tablegen(ATenOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ATenOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRATenOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRATenOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToStd.td)
mlir_tablegen(ATenToStd.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToStdIncGen)

set(LLVM_TARGET_DEFINITIONS ATenPasses.td)
mlir_tablegen(ATenPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRATenPassIncGen)

add_mlir_doc(ATenPasses -gen-pass-doc ConversionPasses ./)

set(LLVM_TARGET_DEFINITIONS AIRPasses.td)
mlir_tablegen(AIRPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRAIRPassIncGen)