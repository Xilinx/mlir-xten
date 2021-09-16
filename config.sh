CLANG_VER=8
cmake -GNinja \
		-DLLVM_DIR=$1/../peano/lib/cmake/llvm \
		-DMLIR_DIR=$1/../peano/lib/cmake/mlir \
		-DPYTHON_EXECUTABLE=/usr/bin/python3 \
		-DPython3_EXECUTABLE=/usr/bin/python3 \
		-DCMAKE_C_COMPILER=clang-${CLANG_VER} \
		-DCMAKE_CXX_COMPILER=clang++-${CLANG_VER} \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX=$3 \
		-B$1 -H$2
