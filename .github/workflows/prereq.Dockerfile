FROM ubuntu:focal

WORKDIR /build

# install stuff that is needed for compiling LLVM, MLIR and ONNX
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake ninja-build libprotobuf-dev protobuf-compiler 
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get -y install \
        make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils \
        libffi-dev liblzma-dev

# RUN git clone git://github.com/yyuu/pyenv.git .pyenv
# RUN git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

# ENV HOME=/build
# ENV PYENV_ROOT=$HOME/.pyenv
# ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# RUN pyenv install 3.7.0

# RUN pyenv global 3.7.0
# RUN pyenv rehash

RUN apt-get install python3 python3-dev python3-pip --assume-yes

RUN apt-get install clang-8 lld-8 git ninja-build --assume-yes

RUN pip3 install pybind11 numpy

RUN pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
#RUN pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.20.6/cmake-3.20.6-linux-x86_64.tar.gz > cmake.tgz
RUN tar -xvf cmake.tgz
RUN cp -r cmake*x86_64/bin/* /usr/bin
RUN cp -r cmake*x86_64/share/* /usr/share

# first install MLIR in llvm-project
# RUN mkdir bin
# ENV PATH=$PATH:/build/bin
# COPY clone-mlir.sh bin/clone-mlir.sh
# RUN chmod a+x bin/clone-mlir.sh
# RUN clone-mlir.sh

# WORKDIR /build/llvm-project/build
# RUN cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS=mlir \
#    -DLLVM_TARGETS_TO_BUILD="host" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON \
#    -DLLVM_ENABLE_RTTI=ON

# RUN cmake --build . --target -- ${MAKEFLAGS}
# RUN cmake --build . --target check-mlir
