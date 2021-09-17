set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XTenConversion)
add_public_tablegen_target(XTenConversionIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToXTen.td)
mlir_tablegen(ATenToXTen.cpp.inc -gen-rewriters)
add_public_tablegen_target(ATenToXTenIncGen)

#add_mlir_doc(Passes XTenConversionPasses ./ -gen-pass-doc)