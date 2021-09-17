set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ATenConversion)
add_public_tablegen_target(ATenConversionIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToXTen.td)
mlir_tablegen(ATenToXTen.cpp.inc -gen-rewriters)
add_public_tablegen_target(ATenToXTenIncGen)

#add_mlir_doc(Passes AIRConversionPasses ./ -gen-pass-doc)