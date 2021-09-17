add_mlir_library(XTenTransformPasses
  ATenDialectOpStats.cpp
  ATenLoweringPass.cpp
  ATenOpReport.cpp
  LowerToLibATenPass.cpp
  Passes.cpp

  DEPENDS
  XTenTransformIncGen

  LINK_LIBS
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  MLIR
  )
