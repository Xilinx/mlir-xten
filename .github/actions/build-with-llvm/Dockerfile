# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

# Container image that runs your code
FROM ghcr.io/stephenneuendorffer/torch-mlir:main
#FROM ghcr.io/stephenneuendorffer/mlir-xten-llvm:main

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY entrypoint.sh /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]