set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ATenTransform)
add_public_tablegen_target(MLIRATenTransformIncGen)

#add_mlir_doc(Passes ATenTransformPasses ./ -gen-pass-doc)