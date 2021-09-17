set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XTenTransform)
add_public_tablegen_target(XTenTransformIncGen)

#add_mlir_doc(Passes XTenTransformPasses ./ -gen-pass-doc)