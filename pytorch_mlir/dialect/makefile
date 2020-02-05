# (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
include ../../config.make

BUILD_DIR = $(abspath ../../build$(BUILD)/pytorch_mlir/dialect)

CXX = clang++-8
SRCDIR = csrc
INCDIR = include
OBJDIR = ${BUILD_DIR}/obj

LLVM_BUILD=$(abspath ../../build$(BUILD)/peano)
LLVM_BIN=${LLVM_BUILD}/bin
LLVM_PROJECT=$(realpath ../../peano/llvm-project)
LLVM_INCLUDE=${LLVM_PROJECT}/llvm/include
MLIR_INCLUDE=$(LLVM_PROJECT)/llvm/projects/mlir/include

ATEN_DIALECT_SRC_FILES = $(SRCDIR)/ATenDialect.cpp $(SRCDIR)/ATenDialectOpStats.cpp 
ATEN_DIALECT_INC_FILES = $(INCDIR)/ATenDialect.h
ATEN_DIALECT_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(ATEN_DIALECT_SRC_FILES))

ATEN_PASS_SRC_FILES = $(SRCDIR)/ATenLoweringPass.cpp $(SRCDIR)/ATenToStd.cpp \
					  $(SRCDIR)/ATenOpReport.cpp $(SRCDIR)/LivenessReport.cpp \
					  $(SRCDIR)/ATenLayerNamePass.cpp
ATEN_PASS_INC_FILES = $(INCDIR)/ATenLoweringPass.h $(INCDIR)/ATenOpReport.h \
					  $(INCDIR)/LivenessReport.h $(INCDIR)/LivenessReport.h \
					  $(INCDIR)/ATenLayerNamePass.cpp
ATEN_PASS_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(ATEN_PASS_SRC_FILES))

ATEN_TBGEN_FILES = $(BUILD_DIR)/include/ATenOps.h.inc \
				   $(BUILD_DIR)/include/ATenOps.cpp.inc \
					$(BUILD_DIR)/include/ATenOpInterfaces.h.inc \
				   $(BUILD_DIR)/include/ATenOpInterfaces.cpp.inc \
				   $(BUILD_DIR)/include/ATenToStd.cpp.inc

LLVM_CXX_FLAGS = $(shell ${LLVM_BIN}/llvm-config --cppflags)
LLVM_LIBS = -lLLVM-10git
#$(shell ${LLVM_BIN}/llvm-config --libs)
LLVM_SYSTEM_LIBS = $(shell ${LLVM_BIN}/llvm-config --system-libs)
LLVM_LIBDIR = $(shell ${LLVM_BIN}/llvm-config --libdir)
LLVM_LD_FLAGS = $(shell ${LLVM_BIN}/llvm-config --ldflags) -Wl,-rpath=$(LLVM_LIBDIR)

MLIR_CXX_FLAGS = -I${LLVM_INCLUDE} \
					  -I../include \
                 -I$(MLIR_INCLUDE) \
					  -I$(LLVM_BUILD)/projects/mlir/include

MLIR_LD_FLAGS =

CXX_FLAGS = -std=c++14 -O0 -fPIC -g -I$(INCDIR) -I$(BUILD_DIR)/include \
            $(LLVM_CXX_FLAGS) $(MLIR_CXX_FLAGS)
LD_FLAGS = -std=c++14 -O0 -g $(LLVM_LD_FLAGS) $(MLIR_LD_FLAGS)

###################
# rules
###################

TARGETS = $(addprefix $(BUILD_DIR)/,test.exe)

build: $(TARGETS)

export TEST_SRC_PATH = $(realpath ${SRCDIR})
export TEST_BUILD_PATH = $(realpath ${BUILD_DIR})
export PYTHONPATH = ${TEST_BUILD_PATH}:${TEST_SRC_PATH}/../..
export PATH := ${PATH}:$(realpath ${LLVM_BIN})

test: build
	${LLVM_BIN}/llvm-lit -v -s ./test --path=$(TEST_BUILD_PATH)

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
# test.exe
###################

$(BUILD_DIR)/test.exe: $(OBJDIR)/test.o $(BUILD_DIR)/libaten.a makefile
	$(CXX) $< $(BUILD_DIR)/libaten.a  $(LD_FLAGS) -o $@ $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) -lcurses -L$(BUILD_DIR)
