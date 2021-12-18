FROM ghcr.io/stephenneuendorffer/mlir-xten-base:main AS builder

WORKDIR /build

# first install MLIR in llvm-project
RUN mkdir bin
ENV PATH=$PATH:/build/bin
COPY utils/clone-llvm.sh bin/clone-llvm.sh
RUN chmod a+x bin/clone-llvm.sh
RUN clone-llvm.sh

COPY utils/build-llvm.sh bin/build-llvm.sh
RUN chmod a+x bin/build-llvm.sh
RUN build-llvm.sh

# FROM ghcr.io/stephenneuendorffer/mlir-xten-base:main
# WORKDIR /build
# RUN mkdir install
# COPY --from=builder /build/install install
# COPY --from=builder /build/llvm llvm

# RUN cmake --build . --target check-mlir
