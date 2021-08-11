set(LLVM_OPTIONAL_SOURCES
  AcapLoopLoweringPass.cpp
  AcapHerdLoweringPass.cpp
  AcapOutliner.cpp
  ATenDialect.cpp
  ATenDialectOpStats.cpp
  ATenLoweringPass.cpp
  ATenModulePass.cpp
  ATenOpReport.cpp
  ATenSimpleAllocatePass.cpp
  ATenToStd.cpp
  LivenessReport.cpp
  ReturnEliminationPass.cpp
  Util.cpp
  LowerToLibATenPass.cpp
  )

add_mlir_library(ATen
  AcapLoopLoweringPass.cpp
  AcapHerdLoweringPass.cpp
  AcapOutliner.cpp
  ATenDialect.cpp
  ATenDialectOpStats.cpp
  ATenLoweringPass.cpp
  ATenModulePass.cpp
  ATenOpReport.cpp
  ATenSimpleAllocatePass.cpp
  ATenToStd.cpp
  LivenessReport.cpp
  ReturnEliminationPass.cpp
  Util.cpp
  LowerToLibATenPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/dialect/include
  ${PROJECT_BINARY_DIR}/dialect/include

  DEPENDS

  LINK_LIBS
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  MLIR

  AIE
  )
llvm_update_compile_flags(ATen)
