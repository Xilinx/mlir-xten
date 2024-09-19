# (c) Copyright 2024 Advanced Micro Devices, Inc. All Rights reserved.

add_mlir_public_c_api_library(
  XTenCAPI
  Dialects.cpp
  ADDITIONAL_HEADER_DIRS
  ${XTEN_SOURCE_DIR}/include/xten-c/
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  XTenNNDialect
)
