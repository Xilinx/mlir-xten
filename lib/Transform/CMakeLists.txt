add_mlir_library(XTenTransformPasses
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
  MLIRXTenTransformIncGen

  LINK_LIBS
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  MLIR
  )
