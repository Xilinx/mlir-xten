add_mlir_library(XTenDialect
  XTenDialect.cpp
  XTenOps.cpp
  XTenOpWrapper.cpp
  XTenOpStats.cpp

  DEPENDS
  MLIRXTenOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)