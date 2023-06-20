//===- XTenOpWrapper.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_OP_WRAPPER
#define XTEN_OP_WRAPPER

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Arch.h"

// NOTE this could be merged with the ops directly possibly, but would need to both NPCOMP and XTEN
// But we will need the build function here anyway as each op will have different arguments and we
// don't want that in the pattern themselves
// So then not a big addition at the moment, especially with the NPCOMP dependency

// NOTE this may also be an easy to use that on top of ONNX or ATen transparently

// TODO whenever we generate something it will always be generating the partial version of the corresponding convolution?

using namespace mlir;
using namespace mlir::torch;
using namespace xilinx::xten;

namespace xilinx {
    namespace xten {

        class AbsOpWrapper {
        public:
            virtual ~AbsOpWrapper() = 0;
            virtual Operation* getUnderlyingOperation() = 0;
            virtual Value getWeights() = 0;
            virtual Value getInput() = 0;
            virtual Value getPartialInput() = 0;
            virtual std::optional<Value> getBiases() = 0;
            virtual ArrayRef<Value> getBN() = 0;
            virtual unsigned int getF0() = 0;
            virtual unsigned int getF1() = 0;
            virtual unsigned int getStride() = 0;
            virtual bool hasWeights() = 0;
            virtual bool hasBias() = 0;
            virtual bool hasBN() = 0;
            virtual bool isDepthWise() = 0;
            virtual double getKernelEfficiency() = 0;
            //virtual bool hasFusedBN(); // TODO
            //virtual Value getBNWeights();
            //virtual Value getBNBias();
            virtual Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                                       std::optional<Value> bias, std::optional<Value> partialIn, bool firstInPartialChain,
                                       std::optional<ArrayRef<Value>> bn) = 0;
            virtual Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) = 0;
        };

        class Conv2dOpWrapper : public AbsOpWrapper {
        private:
            Conv2dOp conv;
        public:
            Conv2dOpWrapper(Conv2dOp c);
            ~Conv2dOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };

        class PartialConv2dOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dOp conv;
        public:
            PartialConv2dOpWrapper(PartialConv2dOp c);
            ~PartialConv2dOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };

        class Conv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            Conv2dReLUOp conv;
        public:
            Conv2dReLUOpWrapper(Conv2dReLUOp c);
            ~Conv2dReLUOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };

        class PartialConv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dReLUOp conv;
        public:
            PartialConv2dReLUOpWrapper(PartialConv2dReLUOp c);
            ~PartialConv2dReLUOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };

        class Conv2dBatchNormReLUOpWrapper : public AbsOpWrapper {
        private:
            Conv2dBatchNormReLUOp conv;
        public:
            Conv2dBatchNormReLUOpWrapper(Conv2dBatchNormReLUOp c);
            ~Conv2dBatchNormReLUOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };

        class PartialConv2dBatchNormReLUOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dBatchNormReLUOp conv;
        public:
            PartialConv2dBatchNormReLUOpWrapper(PartialConv2dBatchNormReLUOp c);
            ~PartialConv2dBatchNormReLUOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };


        class MaxPool2dOpWrapper : public AbsOpWrapper {
        private:
            Torch::AtenMaxPool2dOp maxpool;
        public:
            MaxPool2dOpWrapper(Torch::AtenMaxPool2dOp c);
            ~MaxPool2dOpWrapper();
            Operation* getUnderlyingOperation() override;
            Value getWeights() override;
            Value getInput() override;
            Value getPartialInput() override;
            std::optional<Value> getBiases() override;
            ArrayRef<Value> getBN() override;
            virtual unsigned int getF0() override;
            virtual unsigned int getF1() override;
            unsigned int getStride() override;
            bool hasWeights() override;
            bool hasBias() override;
            bool hasBN() override;
            bool isDepthWise() override;
            double getKernelEfficiency() override;
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, std::optional<Value> weight,
                               std::optional<Value> bias,std::optional<Value> partialIn, bool firstInPartialChain,
                               std::optional<ArrayRef<Value>> bn) override;
            Operation* wCopy(OpBuilder &builder, unsigned int into, std::optional<TypeRange> resTypes) override;
        };
    }
}

#endif

