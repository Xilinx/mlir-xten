//===- XTenDataflowUtils.h --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_DATAFLOW_UTILS
#define XTEN_DATAFLOW_UTILS

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "xten/Dialect/XTen/XTenOpWrapper.h"
#include "xten/Dialect/XTen/XTenDataflowConsts.h"
#include "xten/Dialect/XTen/XTenOps.h"

using namespace mlir;

namespace xilinx {
    namespace xten {

        enum Split {PSplit, CaSplit, LSplit, WSplit};
        enum SplitType {wSplitType, aSplitType, bSplitType};

        class ModelParams {
        public:
            unsigned int P;
            unsigned int Ca;
            unsigned int L;
            unsigned int W;
            // unsigned int K; // Probably enough as an implicit parameter
            bool lineGranularity;

            ModelParams() {
                P = 1;
                Ca = 1;
                W = 1;
                L = 1;
                lineGranularity = false;
            }

            ModelParams(unsigned int defP, unsigned int defCa, unsigned int defL, unsigned int defW, bool lineGranularity) {
                this->P = defP;
                this->Ca = defCa;
                this->L = defL;
                this->W = defW;
                this->lineGranularity = lineGranularity;
            }

            unsigned int cores() {
                return P * Ca * L * W;
            }

            bool nonZero() {
                return (P != 0) && (Ca != 0) && (L != 0) && (W != 0);
            }

            void print() {
                llvm::outs() << "P: " << this->P << ", Ca: " << this->Ca << ", L: " << this->L << ", W: " << this->W <<
                    ", lineGranularity: " << this->lineGranularity << "\n";
            }
        };

        mlir::torch::Torch::BaseTensorType breakShapeInto(mlir::torch::Torch::BaseTensorType initShape, unsigned int at, unsigned int into);
        mlir::torch::Torch::BaseTensorType mergeShapeInto(mlir::torch::Torch::BaseTensorType initShape, unsigned int at, unsigned int into);

        void splitConstantInto(mlir::arith::ConstantOp op, std::vector<Value> &ops, OpBuilder &builder, Split split, SplitType t, unsigned int into);

        void deleteOpsFrom(std::vector<Operation*> &ops);
        void deleteOpsFrom(std::vector<AbsOpWrapper*> &ops);

        Operation* insertConcat(OpBuilder &builder, Value prevRes, std::vector<Value> &values, unsigned int dim, bool clearPrev);
        void replaceConcat(OpBuilder &builder, ConcatOp concat, std::vector<Value> &nInputs,
                           std::vector<Operation*> &toDelete, unsigned int dim, unsigned int into);

        void insertSplit(OpBuilder &builder, Value prevInput, std::vector<Value> &nInputs, unsigned int dim, unsigned int into);
        void replaceSplit(OpBuilder &builder, SplitOp split, std::vector<Value> &values,
                          std::vector<Operation*> &toDelete, unsigned int dim);


        unsigned int getAttrOrDefault(Operation* op, std::string attrName, unsigned int defVal);
        void printOperationLoc(Operation* op);
    }
}

#endif
