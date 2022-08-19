//===- XTenOpWrapper.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTen/XTenOpWrapper.h"
#include "xten/Dialect/XTen/XTenDataflowConsts.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

// TODO generate this file automatically

using namespace mlir::torch;

namespace xilinx {
    namespace xten {

        AbsOpWrapper::~AbsOpWrapper() {}

        Conv2dOpWrapper::Conv2dOpWrapper(Conv2dOp c) {
            conv = c;
        }

        Conv2dOpWrapper::~Conv2dOpWrapper() {}

        Operation* Conv2dOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> Conv2dOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Optional<Value> Conv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int Conv2dOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int Conv2dOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t, 2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }


        bool Conv2dOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool Conv2dOpWrapper::hasBN() {
            return false;
        }

        bool Conv2dOpWrapper::isDepthWise() {
            //unsigned int groups = this->conv.groups().getDefiningOp<mlir::torch::Torch::IntType>().get();//.value();
          llvm::APInt intT = this->conv.groups().getDefiningOp<mlir::torch::Torch::ConstantIntOp>().value();
            //MLIRContext *context = intT.getContext();
          uint64_t groups = intT.getSExtValue();

            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            uint64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double Conv2dOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* Conv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                            llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                            llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());

            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Value biasVal = bias.hasValue() ? bias.getValue() : this->conv.bias();

            Operation* op = this->getUnderlyingOperation();
            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                                  returnType,
                                                                  input,
                                                                  chainIn,
                                                                  weight.getValue(),
                                                                  biasVal,
                                                                  this->conv.stride(),
                                                                  this->conv.padding(),
                                                                  this->conv.dilation(),
                                                                  this->conv.groups());

                nOp->setAttrs(op->getAttrs());

                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dOp>(builder.getUnknownLoc(),
                                                          returnType,
                                                          input,
                                                          weight.getValue(),
                                                          biasVal,
                                                          this->conv.stride(),
                                                          this->conv.padding(),
                                                          this->conv.dilation(),
                                                          this->conv.groups());

                nOp->setAttrs(op->getAttrs());

                return nOp;
            }
        }

        Operation* Conv2dOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                      resTypes.getValue(),
                                                      this->getInput(),
                                                      nullptr,
                                                      this->getWeights(),
                                                      this->getBiases().value_or(nullptr),
                                                      this->conv.stride(),
                                                      this->conv.padding(),
                                                      this->conv.dilation(),
                                                      this->conv.groups());

            } else {
                op = builder.create<Conv2dOp>(builder.getUnknownLoc(),
                                              this->getUnderlyingOperation()->getResultTypes(),
                                              this->getInput(),
                                              this->getWeights(),
                                              this->getBiases().value_or(nullptr),
                                              this->conv.stride(),
                                              this->conv.padding(),
                                              this->conv.dilation(),
                                              this->conv.groups());
            }

            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        PartialConv2dOpWrapper::PartialConv2dOpWrapper(PartialConv2dOp c) {
            conv = c;
        }

        PartialConv2dOpWrapper::~PartialConv2dOpWrapper() {}

        Operation* PartialConv2dOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> PartialConv2dOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Optional<Value> PartialConv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int PartialConv2dOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int PartialConv2dOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t, 2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));
            return stride[0];
        }

        bool PartialConv2dOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool PartialConv2dOpWrapper::hasBN() {
            return false;
        }

        bool PartialConv2dOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<mlir::arith::ConstantIntOp>().value();
            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            int64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double PartialConv2dOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* PartialConv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                   llvm::Optional<Value> bias, llvm::Optional<Value> partialIn,
                                                   bool firstInPartialChain, llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Value biasVal = bias.hasValue() ? bias.getValue() : this->conv.bias();

            Value chainIn;
            if(partialIn.hasValue()) {
                chainIn = partialIn.getValue();
            } else if(this->conv.PartialIn()){
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = Value();
            }

            Operation* op = this->getUnderlyingOperation();

            Operation* nOp =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                              returnType,
                                                              input,
                                                              chainIn,
                                                              weight.getValue(),
                                                              biasVal,
                                                              this->conv.stride(),
                                                              this->conv.padding(),
                                                              this->conv.dilation(),
                                                              this->conv.groups());

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                     resTypes.getValue(),
                                                     this->getInput(),
                                                     this->conv.PartialIn(),
                                                     this->getWeights(),
                                                     this->getBiases().value_or(nullptr),
                                                     this->conv.stride(),
                                                     this->conv.padding(),
                                                     this->conv.dilation(),
                                                     this->conv.groups());
            } else {
                op = builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                     this->getUnderlyingOperation()->getResultTypes(),
                                                     this->getInput(),
                                                     this->conv.PartialIn(),
                                                     this->getWeights(),
                                                     this->getBiases().value_or(nullptr),
                                                     this->conv.stride(),
                                                     this->conv.padding(),
                                                     this->conv.dilation(),
                                                     this->conv.groups());
            }


            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        Conv2dReLUOpWrapper::Conv2dReLUOpWrapper(Conv2dReLUOp c) {
            conv = c;
        }

        Conv2dReLUOpWrapper::~Conv2dReLUOpWrapper() {}

        Operation* Conv2dReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> Conv2dReLUOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Optional<Value> Conv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dReLUOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dReLUOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int Conv2dReLUOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int Conv2dReLUOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t,2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }

        bool Conv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dReLUOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool Conv2dReLUOpWrapper::hasBN() {
            return false;
        }

        bool Conv2dReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<mlir::arith::ConstantIntOp>().value();
            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            int64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double Conv2dReLUOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* Conv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());

            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Operation* op = this->getUnderlyingOperation();
            Value biasVal = (bias.hasValue()) ? bias.getValue() : this->conv.bias();

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                      returnType,
                                                                      input,
                                                                      chainIn,
                                                                      weight.getValue(),
                                                                      biasVal,
                                                                      this->conv.stride(),
                                                                      this->conv.padding(),
                                                                      this->conv.dilation(),
                                                                      this->conv.groups());

                nOp->setAttrs(op->getAttrs());
                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                              returnType,
                                                              input,
                                                              weight.getValue(),
                                                              biasVal,
                                                              this->conv.stride(),
                                                              this->conv.padding(),
                                                              this->conv.dilation(),
                                                              this->conv.groups());

                nOp->setAttrs(op->getAttrs());
                return nOp;
            }
        }

        Operation* Conv2dReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                          resTypes.getValue(),
                                                          this->getInput(),
                                                          Value(),
                                                          this->getWeights(),
                                                          this->getBiases().value_or(nullptr),
                                                          this->conv.stride(),
                                                          this->conv.padding(),
                                                          this->conv.dilation(),
                                                          this->conv.groups());
            } else {
                op = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                  this->getUnderlyingOperation()->getResultTypes(),
                                                  this->getInput(),
                                                  this->getWeights(),
                                                  this->getBiases().value_or(nullptr),
                                                  this->conv.stride(),
                                                  this->conv.padding(),
                                                  this->conv.dilation(),
                                                  this->conv.groups());
            }


            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        PartialConv2dReLUOpWrapper::PartialConv2dReLUOpWrapper(PartialConv2dReLUOp c) {
            conv = c;
        }

        PartialConv2dReLUOpWrapper::~PartialConv2dReLUOpWrapper() {}

        Operation* PartialConv2dReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> PartialConv2dReLUOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Optional<Value> PartialConv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dReLUOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dReLUOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int PartialConv2dReLUOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int PartialConv2dReLUOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t,2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }

        bool PartialConv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dReLUOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool PartialConv2dReLUOpWrapper::hasBN() {
            return false;
        }

        bool PartialConv2dReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<mlir::arith::ConstantIntOp>().value();
            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            int64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double PartialConv2dReLUOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* PartialConv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                       llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                       llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                       llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());

            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Value chainIn;
            if(partialIn.hasValue()) {
                chainIn = partialIn.getValue();
            } else if(this->conv.PartialIn()){
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = Value();
            }

            Value biasVal = bias.hasValue() ? bias.getValue() : this->conv.bias();

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                                 returnType,
                                                                 input,
                                                                 chainIn,
                                                                 weight.getValue(),
                                                                 biasVal,
                                                                 this->conv.stride(),
                                                                 this->conv.padding(),
                                                                 this->conv.dilation(),
                                                                 this->conv.groups());
            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                         resTypes.getValue(),
                                                         this->getInput(),
                                                         this->conv.PartialIn(),
                                                         this->getWeights(),
                                                         this->getBiases().value_or(nullptr),
                                                         this->conv.stride(),
                                                         this->conv.padding(),
                                                         this->conv.dilation(),
                                                         this->conv.groups());
            } else {
                op = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                         this->getUnderlyingOperation()->getResultTypes(),
                                                         this->getInput(),
                                                         this->conv.PartialIn(),
                                                         this->getWeights(),
                                                         this->getBiases().value_or(nullptr),
                                                         this->conv.stride(),
                                                         this->conv.padding(),
                                                         this->conv.dilation(),
                                                         this->conv.groups());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        PartialConv2dBatchNormReLUOpWrapper::PartialConv2dBatchNormReLUOpWrapper(PartialConv2dBatchNormReLUOp c) {
            conv = c;
        }

        PartialConv2dBatchNormReLUOpWrapper::~PartialConv2dBatchNormReLUOpWrapper() {}

        Operation* PartialConv2dBatchNormReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> PartialConv2dBatchNormReLUOpWrapper::getBN() {
            return ArrayRef<Value>({this->conv.bn_weight(), this->conv.bn_bias(), this->conv.running_mean(), this->conv.running_var()});
        }

        Optional<Value> PartialConv2dBatchNormReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dBatchNormReLUOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int PartialConv2dBatchNormReLUOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int PartialConv2dBatchNormReLUOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t,2 > stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }

        bool PartialConv2dBatchNormReLUOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dBatchNormReLUOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool PartialConv2dBatchNormReLUOpWrapper::hasBN() {
            return true;
        }

        bool PartialConv2dBatchNormReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<mlir::arith::ConstantIntOp>().value();
            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            int64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double PartialConv2dBatchNormReLUOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* PartialConv2dBatchNormReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                                llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                                llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bn.hasValue());

            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Value chainIn;
            if(partialIn.hasValue()) {
                chainIn = partialIn.getValue();
            } else if(this->conv.PartialIn()){
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = Value();
            }

            Value biasVal = bias.hasValue() ? bias.getValue() : this->conv.bias();

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                          returnType,
                                                                          input,
                                                                          chainIn,
                                                                          weight.getValue(),
                                                                          biasVal,
                                                                          this->conv.stride(),
                                                                          this->conv.padding(),
                                                                          this->conv.dilation(),
                                                                          this->conv.groups(),
                                                                          bn.getValue()[0],
                                                                          bn.getValue()[1],
                                                                          bn.getValue()[2],
                                                                          bn.getValue()[3],
                                                                          this->conv.training(),
                                                                          this->conv.momentum(),
                                                                          this->conv.eps());

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dBatchNormReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;
            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                  resTypes.getValue(),
                                                                  this->getInput(),
                                                                  this->conv.PartialIn(),
                                                                  this->getWeights(),
                                                                  this->getBiases().value_or(nullptr),
                                                                  this->conv.stride(),
                                                                  this->conv.padding(),
                                                                  this->conv.dilation(),
                                                                  this->conv.groups(),
                                                                  this->conv.bn_weight(),
                                                                  this->conv.bn_bias(),
                                                                  this->conv.running_mean(),
                                                                  this->conv.running_var(),
                                                                  this->conv.training(),
                                                                  this->conv.momentum(),
                                                                  this->conv.eps());
            } else {
                op = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                  this->getUnderlyingOperation()->getResultTypes(),
                                                                  this->getInput(),
                                                                  this->conv.PartialIn(),
                                                                  this->getWeights(),
                                                                  this->getBiases().value_or(nullptr),
                                                                  this->conv.stride(),
                                                                  this->conv.padding(),
                                                                  this->conv.dilation(),
                                                                  this->conv.groups(),
                                                                  this->conv.bn_weight(),
                                                                  this->conv.bn_bias(),
                                                                  this->conv.running_mean(),
                                                                  this->conv.running_var(),
                                                                  this->conv.training(),
                                                                  this->conv.momentum(),
                                                                  this->conv.eps());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }


        Conv2dBatchNormReLUOpWrapper::Conv2dBatchNormReLUOpWrapper(Conv2dBatchNormReLUOp c) {
            conv = c;
        }

        Conv2dBatchNormReLUOpWrapper::~Conv2dBatchNormReLUOpWrapper() {}

        Operation* Conv2dBatchNormReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dBatchNormReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> Conv2dBatchNormReLUOpWrapper::getBN() {
            return ArrayRef<Value>({this->conv.bn_weight(), this->conv.bn_bias(), this->conv.running_mean(), this->conv.running_var()});
        }

        Optional<Value> Conv2dBatchNormReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dBatchNormReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dBatchNormReLUOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dBatchNormReLUOpWrapper::getF0() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F0_LOC];
        }

        unsigned int Conv2dBatchNormReLUOpWrapper::getF1() {
            return this->conv.weight().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>().getSizes()[F1_LOC];
        }

        unsigned int Conv2dBatchNormReLUOpWrapper::getStride() {
            Value s = this->conv.stride();
            SmallVector<int64_t,2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }

        bool Conv2dBatchNormReLUOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dBatchNormReLUOpWrapper::hasBias() {
            return this->getBiases().hasValue();
        }

        bool Conv2dBatchNormReLUOpWrapper::hasBN() {
            return true;
        }

        bool Conv2dBatchNormReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<mlir::arith::ConstantIntOp>().value();
            mlir::torch::Torch::BaseTensorType aShape = this->conv.input().getType().dyn_cast<mlir::torch::Torch::BaseTensorType>();
            ArrayRef<int64_t> aShapeAR = aShape.getSizes();

            int64_t C = aShapeAR[C_LOC];

            return groups == C;
        }

        double Conv2dBatchNormReLUOpWrapper::getKernelEfficiency() {
            if(this->isDepthWise()) {
                return 0.30;
            } else {
                return 0.90;
            }
        }

        Operation* Conv2dBatchNormReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                                llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                                llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bn.hasValue());

            if(this->hasBias()) {
                assert(bias.hasValue());
            }

            Value biasVal = bias.hasValue() ? bias.getValue() : this->conv.bias();

            Operation* op = this->getUnderlyingOperation();
            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                               returnType,
                                                                               input,
                                                                               chainIn,
                                                                               weight.getValue(),
                                                                               biasVal,
                                                                               this->conv.stride(),
                                                                               this->conv.padding(),
                                                                               this->conv.dilation(),
                                                                               this->conv.groups(),
                                                                               bn.getValue()[0],
                                                                               bn.getValue()[1],
                                                                               bn.getValue()[2],
                                                                               bn.getValue()[3],
                                                                               this->conv.training(),
                                                                               this->conv.momentum(),
                                                                               this->conv.eps());
                nOp->setAttrs(op->getAttrs());

                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                       returnType,
                                                                       input,
                                                                       weight.getValue(),
                                                                       biasVal,
                                                                       this->conv.stride(),
                                                                       this->conv.padding(),
                                                                       this->conv.dilation(),
                                                                       this->conv.groups(),
                                                                       bn.getValue()[0],
                                                                       bn.getValue()[1],
                                                                       bn.getValue()[2],
                                                                       bn.getValue()[3],
                                                                       this->conv.training(),
                                                                       this->conv.momentum(),
                                                                       this->conv.eps());

                nOp->setAttrs(op->getAttrs());

                return nOp;
            }
        }

        Operation* Conv2dBatchNormReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                   resTypes.getValue(),
                                                                   this->getInput(),
                                                                   Value(),
                                                                   this->getWeights(),
                                                                   this->getBiases().value_or(nullptr),
                                                                   this->conv.stride(),
                                                                   this->conv.padding(),
                                                                   this->conv.dilation(),
                                                                   this->conv.groups(),
                                                                   this->conv.bn_weight(),
                                                                   this->conv.bn_bias(),
                                                                   this->conv.running_mean(),
                                                                   this->conv.running_var(),
                                                                   this->conv.training(),
                                                                   this->conv.momentum(),
                                                                   this->conv.eps());
            } else {
                op = builder.create<Conv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                           this->getUnderlyingOperation()->getResultTypes(),
                                                           this->getInput(),
                                                           this->getWeights(),
                                                           this->getBiases().value_or(nullptr),
                                                           this->conv.stride(),
                                                           this->conv.padding(),
                                                           this->conv.dilation(),
                                                           this->conv.groups(),
                                                           this->conv.bn_weight(),
                                                           this->conv.bn_bias(),
                                                           this->conv.running_mean(),
                                                           this->conv.running_var(),
                                                           this->conv.training(),
                                                           this->conv.momentum(),
                                                           this->conv.eps());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        MaxPool2dOpWrapper::MaxPool2dOpWrapper(Torch::AtenMaxPool2dOp mp) {
            maxpool = mp;
        }

        MaxPool2dOpWrapper::~MaxPool2dOpWrapper() {}


        Operation* MaxPool2dOpWrapper::getUnderlyingOperation() {
            return maxpool.getOperation();
        }

        Value MaxPool2dOpWrapper::getWeights() {
            return Value();
        }

        Optional<Value> MaxPool2dOpWrapper::getBiases() {
            return Optional<Value>{};
        }

        unsigned int MaxPool2dOpWrapper::getF0() {
            Value ks = this->maxpool.kernel_size();
            SmallVector<int64_t,2> kernel_size;
            matchPattern(ks, Torch::m_TorchConstantIntList(kernel_size));

            return kernel_size[0];
        }

        unsigned int MaxPool2dOpWrapper::getF1() {
            Value ks = this->maxpool.kernel_size();
            SmallVector<int64_t,2> kernel_size;
            matchPattern(ks, Torch::m_TorchConstantIntList(kernel_size));

            return kernel_size[1];
        }

        unsigned int MaxPool2dOpWrapper::getStride() {
            Value s = this->maxpool.stride();
            SmallVector<int64_t,2> stride;
            matchPattern(s, Torch::m_TorchConstantIntList(stride));

            return stride[0];
        }

        Value MaxPool2dOpWrapper::getInput() {
            return this->maxpool.self();
        }

        Value MaxPool2dOpWrapper::getPartialInput() {
            return Value();
        }

        ArrayRef<Value> MaxPool2dOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        bool MaxPool2dOpWrapper::hasWeights() {
            return false;
        }

        bool MaxPool2dOpWrapper::hasBias() {
            return false;
        }

        bool MaxPool2dOpWrapper::isDepthWise() {
            return true;
        }

        bool MaxPool2dOpWrapper::hasBN() {
            return false;
        }

        double MaxPool2dOpWrapper::getKernelEfficiency() {
            return 0.25;
        }

        Operation* MaxPool2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                          llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                          llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                          llvm::Optional<ArrayRef<Value>> bn) {
            assert(!weight.hasValue());
            assert(!bias.hasValue());
            assert(!firstInPartialChain);
            assert(!partialIn.hasValue());

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp =  builder.create<Torch::AtenMaxPool2dOp>(builder.getUnknownLoc(), returnType, input,
                                                                     this->maxpool.kernel_size(),
                                                                     this->maxpool.stride(),
                                                                     this->maxpool.padding(),
                                                                     this->maxpool.dilation(),
                                                                     this->maxpool.ceil_mode());

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* MaxPool2dOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> typeRes) {
            assert(!typeRes.hasValue());

            Operation* op =  builder.create<Torch::AtenMaxPool2dOp>(builder.getUnknownLoc(),
                                                                    this->getUnderlyingOperation()->getResultTypes(),
                                                                    this->getInput(),
                                                                    this->maxpool.kernel_size(),
                                                                    this->maxpool.stride(),
                                                                    this->maxpool.padding(),
                                                                    this->maxpool.dilation(),
                                                                    this->maxpool.ceil_mode());

            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

    }
}

