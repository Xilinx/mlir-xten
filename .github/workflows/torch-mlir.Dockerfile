FROM ghcr.io/stephenneuendorffer/mlir-xten-llvm:main AS builder

WORKDIR /build

# first install MLIR in llvm-project
ENV PATH=$PATH:/build/bin
COPY utils/build-torch-mlir.sh bin/build-torch-mlir.sh
RUN chmod a+x bin/build-torch-mlir.sh
RUN build-torch-mlir.sh

# FROM ghcr.io/stephenneuendorffer/mlir-xten-base:main
# WORKDIR /build
# RUN mkdir install
# COPY --from=builder /build/install install
# COPY --from=builder /build/llvm llvm

# RUN cmake --build . --target check-mlir
