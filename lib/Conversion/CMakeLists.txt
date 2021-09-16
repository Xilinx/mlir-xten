
add_mlir_library(ATenConversionPasses
ATenToXTenPass.cpp
Passes.cpp
XTenToAffine.cpp
XTenToLinalg.cpp

DEPENDS
ATenConversionIncGen
ATenToXTenIncGen


LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
)