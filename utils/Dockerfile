# Container image that runs your code
FROM ghcr.io/stephenneuendorffer/mlir-xten-base:main

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY build-llvm.sh /build/build-llvm.sh
COPY clone-llvm.sh /build/clone-llvm.sh

WORKDIR /build
RUN /build/clone-llvm.sh
RUN /build/build-llvm.sh
