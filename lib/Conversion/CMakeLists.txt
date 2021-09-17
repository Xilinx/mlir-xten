
add_mlir_library(XTenConversionPasses
ATenToXTenPass.cpp
Passes.cpp
XTenToAffine.cpp
XTenToLinalg.cpp

DEPENDS
XTenConversionIncGen
ATenToXTenIncGen

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
)