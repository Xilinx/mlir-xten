add_mlir_library(ATenTransformPasses
  AcapLoopLoweringPass.cpp
  AcapHerdLoweringPass.cpp
  ATenDialectOpStats.cpp
  ATenLoweringPass.cpp
  ATenOpReport.cpp
  ATenSimpleAllocatePass.cpp
  LivenessReport.cpp
  LowerToLibATenPass.cpp
  Passes.cpp

  DEPENDS
  MLIRATenTransformIncGen

  LINK_LIBS
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  MLIR
  )
