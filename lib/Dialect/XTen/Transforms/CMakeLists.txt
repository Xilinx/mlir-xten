add_mlir_library(XTenTransforms
  XTenDataflowExplorer.cpp
  XTenDataflowPass.cpp
  XTenDataflowUtils.cpp
  XTenNamePass.cpp
  XTenPasses.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)