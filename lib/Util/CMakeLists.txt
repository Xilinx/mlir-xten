add_mlir_library(ATenUtil
  Util.cpp
  Outliner.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)