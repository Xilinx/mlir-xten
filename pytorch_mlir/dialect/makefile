include ../../config.make

BUILD_DIR = $(abspath ../../build$(BUILD)/pytorch_mlir/dialect)

CXX = clang++-8
SRCDIR = csrc
INCDIR = include
OBJDIR = ${BUILD_DIR}/obj

PYTHON_PROOT=/usr
# Conda stuffs this in a non-standard place.
# A proper configuration tool would be nice for this.
PYBIND11_INCLUDE_CONFIG-container=/opt/conda/include/python3.6m
PYBIND11_INCLUDE_CONFIG=/usr/local/include/python3.6
PYBIND11_INCLUDE = ${PYBIND11_INCLUDE_CONFIG${BUILD}}
LLVM_BUILD=$(abspath ../../build$(BUILD)/peano)
LLVM_BIN=${LLVM_BUILD}/bin
LLVM_PROJECT=$(realpath ../../peano/llvm-project)
LLVM_INCLUDE=${LLVM_PROJECT}/llvm/include
MLIR_INCLUDE=$(LLVM_PROJECT)/llvm/projects/mlir/include

ATEN_DIALECT_SRC_FILES = $(SRCDIR)/ATenDialect.cpp
ATEN_DIALECT_INC_FILES = $(INCDIR)/ATenDialect.h
ATEN_DIALECT_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(ATEN_DIALECT_SRC_FILES))

ATEN_PASS_SRC_FILES = $(SRCDIR)/ATenLoweringPass.cpp $(SRCDIR)/ATenToStd.cpp $(SRCDIR)/ForwardPathReport.cpp
ATEN_PASS_INC_FILES = $(INCDIR)/ATenLoweringPass.h $(INCDIR)/ForwardPathReport.h
ATEN_PASS_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(ATEN_PASS_SRC_FILES))

ATEN_TBGEN_FILES = $(BUILD_DIR)/include/ATenOps.h.inc \
				   $(BUILD_DIR)/include/ATenOps.cpp.inc \
					$(BUILD_DIR)/include/ATenOpInterfaces.h.inc \
				   $(BUILD_DIR)/include/ATenOpInterfaces.cpp.inc \
				   $(BUILD_DIR)/include/ATenToStd.cpp.inc

LLVM_CXX_FLAGS = $(shell ${LLVM_BIN}/llvm-config --cppflags)
LLVM_LIBS = $(shell ${LLVM_BIN}/llvm-config --libs)
LLVM_LIBDIR = $(shell ${LLVM_BIN}/llvm-config --libdir)
LLVM_LD_FLAGS = $(shell ${LLVM_BIN}/llvm-config --ldflags) -Wl,-rpath=$(LLVM_LIBDIR)

MLIR_CXX_FLAGS = -I${LLVM_INCLUDE} \
					  -I../include \
                 -I$(MLIR_INCLUDE) \
					  -I$(LLVM_BUILD)/projects/mlir/include

MLIR_LD_FLAGS =

CXX_FLAGS = -std=c++14 -O0 -fPIC -g -I$(INCDIR) -I$(BUILD_DIR)/include \
            $(LLVM_CXX_FLAGS) $(MLIR_CXX_FLAGS)
LD_FLAGS = -std=c++14 -O0 -g $(LLVM_LD_FLAGS) $(MLIR_LD_FLAGS) -L$(PYTHON_PROOT)/lib

###################
# rules
###################

TARGETS = $(addprefix $(BUILD_DIR)/,test.exe pybind.so pybind_aten.so)

build: $(TARGETS)

export TEST_SRC_PATH = $(realpath ${SRCDIR})
export TEST_BUILD_PATH = $(realpath ${BUILD_DIR})
export PYTHONPATH = ${TEST_BUILD_PATH}:${TEST_SRC_PATH}/../..
export PATH := ${PATH}:$(realpath ${LLVM_BIN})

test: build
	${LLVM_BIN}/llvm-lit -v -s ../mlir_export/test

clean:
	rm -rf $(BUILD_DIR) $(TARGETS)

###################
# table gen rules
###################

$(BUILD_DIR)/include/ATenOps.h.inc: $(INCDIR)/aten.td
	mkdir -p $(BUILD_DIR)/include && \
	${LLVM_BIN}/mlir-tblgen -I$(MLIR_INCLUDE) $< --gen-op-decls > $@

$(BUILD_DIR)/include/ATenOps.cpp.inc: $(INCDIR)/aten.td
	mkdir -p $(BUILD_DIR)/include && \
	${LLVM_BIN}/mlir-tblgen -I$(MLIR_INCLUDE) $< --gen-op-defs > $@

$(BUILD_DIR)/include/ATenOpInterfaces.h.inc: $(INCDIR)/aten.td
	mkdir -p $(BUILD_DIR)/include && \
	${LLVM_BIN}/mlir-tblgen -I$(MLIR_INCLUDE) $< --gen-op-interface-decls > $@

$(BUILD_DIR)/include/ATenOpInterfaces.cpp.inc: $(INCDIR)/aten.td
	mkdir -p $(BUILD_DIR)/include && \
	${LLVM_BIN}/mlir-tblgen -I$(MLIR_INCLUDE) $< --gen-op-interface-defs > $@

$(BUILD_DIR)/include/ATenToStd.cpp.inc: $(INCDIR)/ATenToStd.td
	mkdir -p $(BUILD_DIR)/include && \
	${LLVM_BIN}/mlir-tblgen -I$(INCDIR) -I$(MLIR_INCLUDE) $< --gen-rewriters > $@


###################
# cpp rules
###################

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(ATEN_TBGEN_FILES)
	mkdir -p $(OBJDIR) && \$(CXX) $(CXX_FLAGS) -fPIC -fno-rtti -c -o $@ $<

$(BUILD_DIR)/libaten.a: $(ATEN_DIALECT_OBJ_FILES) $(ATEN_PASS_OBJ_FILES)
	ar rvs $@ $^

###################
# pybind11 rules
###################

MLIR_PYBIND_DIR=$(LLVM_PROJECT)/llvm/projects/mlir/bindings/python

$(BUILD_DIR)/pybind.so: $(MLIR_PYBIND_DIR)/pybind.cpp makefile
	$(CXX) -std=c++14 -O3 -shared -fPIC -fpermissive \
		$(LLVM_CXX_FLAGS) $(MLIR_CXX_FLAGS) \
		-I$(PYTHON_PROOT)/lib/python3.6/site-packages/torch/include \
		-I$(PYTHON_PROOT)/include/python3.6 \
		-I$(PYBIND11_INCLUDE) \
		-o $@ $< \
		$(LLVM_LD_FLAGS) \
		-lLLVM -Wl,-rpath=$(BUILD_DIR)

$(BUILD_DIR)/pybind_aten.so: $(SRCDIR)/pybind_aten.cpp $(MLIR_PYBIND_DIR)/pybind.cpp $(BUILD_DIR)/libaten.a makefile
	$(CXX) -std=c++14 -O3 -shared -fPIC -fpermissive \
		-I$(INCDIR) -I$(BUILD_DIR)/include $(LLVM_CXX_FLAGS) $(MLIR_CXX_FLAGS) \
		-I$(PYTHON_PROOT)/lib/python3.6/site-packages/torch/include \
		-I$(PYTHON_PROOT)/include/python3.6 \
		-I$(PYBIND11_INCLUDE) \
		-o $@ $< \
		$(LLVM_LD_FLAGS) \
		$(BUILD_DIR)/libaten.a \
		-lLLVM -Wl,-rpath=$(BUILD_DIR)

###################
# test.exe
###################

$(BUILD_DIR)/test.exe: $(OBJDIR)/test.o $(BUILD_DIR)/libaten.a makefile
	$(CXX) $< $(BUILD_DIR)/libaten.a  $(LD_FLAGS) -o $@ -lLLVM  -lpthread -lxml2 -lcurses -L$(BUILD_DIR)
