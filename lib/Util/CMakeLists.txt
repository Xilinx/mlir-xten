add_mlir_library(XTenUtil
  Util.cpp
  Outliner.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)