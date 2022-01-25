
# Intended to be run from the toplevel directory

echo ${PAT_TOKEN} | docker login ghcr.io -u stephenn --password-stdin
docker build -f .github/workflows/prereq.Dockerfile -t ghcr.io/stephenneuendorffer/mlir-xten-base:main .
docker push ghcr.io/stephenneuendorffer/mlir-xten-base:main
docker build -f .github/workflows/llvmmlir.Dockerfile -t ghcr.io/stephenneuendorffer/mlir-xten-llvm:main .
docker push ghcr.io/stephenneuendorffer/mlir-xten-llvm:main
docker build -f .github/workflows/torch-mlir.Dockerfile -t ghcr.io/stephenneuendorffer/torch-mlir:main .
docker push ghcr.io/stephenneuendorffer/torch-mlir:main