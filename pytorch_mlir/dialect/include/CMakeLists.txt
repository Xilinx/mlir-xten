include_directories(${PROJECT_SOURCE_DIR}/dialect/include)
include_directories(${NPCOMP_SOURCE_DIR}/include)

#set(LLVM_TARGET_DEFINITIONS ONNXToAIR.td)
#mlir_tablegen(ONNXToAIR.cpp.inc -gen-rewriters)
#add_public_tablegen_target(MLIRONNXToAIRIncGen)

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