#ifndef AIR_OPS_SPLITTER
#define AIR_OPS_SPLITTER

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "XTenOps.h"
#include "AIRDialect.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "Arch.h"

// NOTE this could be merged with the ops directly possibly, but would need to both NPCOMP and AIR
// But we will need the build function here anyway as each op will have different arguments and we
// don't want that in the pattern themselves
// So then not a big addition at the moment, especially with the NPCOMP dependency

// NOTE this may also be an easy to use that on top of ONNX or ATen transparently

// TODO whenever we generate something it will always be generating the partial version of the corresponding convolution?

using namespace mlir;
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
            virtual Value getBiases() = 0;
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
            virtual Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                       llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                       llvm::Optional<ArrayRef<Value>> bn) = 0;
            virtual Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) = 0;
        };

        class Conv2dOpWrapper : public AbsOpWrapper {
        private:
            Conv2dOp conv;
        public:
            Conv2dOpWrapper(Conv2dOp c);
            ~Conv2dOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };

        class PartialConv2dOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dOp conv;
        public:
            PartialConv2dOpWrapper(PartialConv2dOp c);
            ~PartialConv2dOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };

        class Conv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            Conv2dReLUOp conv;
        public:
            Conv2dReLUOpWrapper(Conv2dReLUOp c);
            ~Conv2dReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };

        class PartialConv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dReLUOp conv;
        public:
            PartialConv2dReLUOpWrapper(PartialConv2dReLUOp c);
            ~PartialConv2dReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };

        class Conv2dBatchNormReLUOpWrapper : public AbsOpWrapper {
        private:
            Conv2dBatchNormReLUOp conv;
        public:
            Conv2dBatchNormReLUOpWrapper(Conv2dBatchNormReLUOp c);
            ~Conv2dBatchNormReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };

        class PartialConv2dBatchNormReLUOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dBatchNormReLUOp conv;
        public:
            PartialConv2dBatchNormReLUOpWrapper(PartialConv2dBatchNormReLUOp c);
            ~PartialConv2dBatchNormReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };


        class MaxPool2dWithIndicesOpWrapper : public AbsOpWrapper {
        private:
            mlir::NPCOMP::aten::MaxPool2dWithIndicesOp maxpool;
        public:
            MaxPool2dWithIndicesOpWrapper(mlir::NPCOMP::aten::MaxPool2dWithIndicesOp c);
            ~MaxPool2dWithIndicesOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getPartialInput();
            Value getBiases();
            ArrayRef<Value> getBN();
            virtual unsigned int getF0();
            virtual unsigned int getF1();
            unsigned int getStride();
            bool hasWeights();
            bool hasBias();
            bool hasBN();
            bool isDepthWise();
            double getKernelEfficiency();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain,
                               llvm::Optional<ArrayRef<Value>> bn);
            Operation* wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes);
        };
    }
}

#endif

