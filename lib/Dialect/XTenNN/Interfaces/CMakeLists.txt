add_mlir_dialect_library(XTenNNInterfaces
        EnclaveOpInterface.cpp

    DEPENDS
        XTenNNIncGen

    LINK_LIBS PUBLIC
        MLIRIR
)