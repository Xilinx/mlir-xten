include_directories(${PROJECT_SOURCE_DIR}/dialect)

add_mlir_dialect(ATen aten)
set(LLVM_TARGET_DEFINITIONS ATen.td)
mlir_tablegen(ATenEnums.h.inc -gen-enum-decls)
mlir_tablegen(ATenEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRATenEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ATenAcapFusion.td)
mlir_tablegen(ATenAcapFusion.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenAcapFusionIncGen)

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