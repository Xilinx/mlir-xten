# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(aten-opt aten-opt.cpp)
llvm_update_compile_flags(aten-opt)

set(LIBS
  ATenConversionPasses
  ATenTransformPasses
  XTenDialect
  XTenTransforms
  ATenUtil
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  NPCOMPTypingCPA
  AIE
  MLIR
  )

target_link_libraries(aten-opt PRIVATE ${LIBS})
