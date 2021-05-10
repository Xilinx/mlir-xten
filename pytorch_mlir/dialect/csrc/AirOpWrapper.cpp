#include "AirOpWrapper.h"
#include "AirDataflowConsts.h"

#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"

namespace xilinx {
    namespace air {

        void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
            if (auto co = op.getDefiningOp<NPCOMP::aten::ConstantOp>()) {
                DenseElementsAttr a = co->template getAttrOfType<DenseElementsAttr>("value");
                for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
            }
            else if (auto co = op.getDefiningOp<NPCOMP::Basicpy::BuildListOp>()) {
                for (auto o : op.getDefiningOp()->getOperands())
                    v.push_back(o.template getDefiningOp<ConstantIntOp>().getValue());
            }
        }

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

        Value Conv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        unsigned int Conv2dOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool Conv2dOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dOpWrapper::isDepthWise() {
            return false;
        }

        Operation* Conv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                            llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                return builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                       returnType,
                                                       input,
                                                       chainIn,
                                                       weight.getValue(),
                                                       bias.getValue(),
                                                       this->conv.stride(),
                                                       this->conv.padding(),
                                                       this->conv.dilation(),
                                                       this->conv.transposed(),
                                                       this->conv.output_padding(),
                                                       this->conv.groups());
            } else {
                return builder.create<Conv2dOp>(builder.getUnknownLoc(),
                                                returnType,
                                                input,
                                                weight.getValue(),
                                                bias.getValue(),
                                                this->conv.stride(),
                                                this->conv.padding(),
                                                this->conv.dilation(),
                                                this->conv.transposed(),
                                                this->conv.output_padding(),
                                                this->conv.groups());
            }
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

        Value PartialConv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        unsigned int PartialConv2dOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool PartialConv2dOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dOpWrapper::isDepthWise() {
            return false;
        }

        Operation* PartialConv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                   llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            return builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                   returnType,
                                                   input,
                                                   chainIn,
                                                   weight.getValue(),
                                                   bias.getValue(),
                                                   this->conv.stride(),
                                                   this->conv.padding(),
                                                   this->conv.dilation(),
                                                   this->conv.transposed(),
                                                   this->conv.output_padding(),
                                                   this->conv.groups());
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

        Value Conv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        unsigned int Conv2dReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool Conv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dReLUOpWrapper::isDepthWise() {
            return false;
        }

        Operation* Conv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                return builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                           returnType,
                                                           input,
                                                           chainIn,
                                                           weight.getValue(),
                                                           bias.getValue(),
                                                           this->conv.stride(),
                                                           this->conv.padding(),
                                                           this->conv.dilation(),
                                                           this->conv.transposed(),
                                                           this->conv.output_padding(),
                                                           this->conv.groups());
            } else {
                return builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                    returnType,
                                                    input,
                                                    weight.getValue(),
                                                    bias.getValue(),
                                                    this->conv.stride(),
                                                    this->conv.padding(),
                                                    this->conv.dilation(),
                                                    this->conv.transposed(),
                                                    this->conv.output_padding(),
                                                    this->conv.groups());
            }
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

        Value PartialConv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        unsigned int PartialConv2dReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool PartialConv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dReLUOpWrapper::isDepthWise() {
            return false;
        }

        Operation* PartialConv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                       llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                       llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            return builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                       returnType,
                                                       input,
                                                       chainIn,
                                                       weight.getValue(),
                                                       bias.getValue(),
                                                       this->conv.stride(),
                                                       this->conv.padding(),
                                                       this->conv.dilation(),
                                                       this->conv.transposed(),
                                                       this->conv.output_padding(),
                                                       this->conv.groups());
        }


        MaxPool2dWithIndicesOpWrapper::MaxPool2dWithIndicesOpWrapper(mlir::NPCOMP::aten::MaxPool2dWithIndicesOp mp) {
            maxpool = mp;
        }

        MaxPool2dWithIndicesOpWrapper::~MaxPool2dWithIndicesOpWrapper() {}


        Operation* MaxPool2dWithIndicesOpWrapper::getUnderlyingOperation() {
            return maxpool.getOperation();
        }

        Value MaxPool2dWithIndicesOpWrapper::getWeights() {
            return Value();
        }

        Value MaxPool2dWithIndicesOpWrapper::getBiases() {
            return Value();
        }

        unsigned int MaxPool2dWithIndicesOpWrapper::getKernelSize() {
            Value ks = this->maxpool.kernel_size();
            std::vector<int64_t> kernel_size;
            unpack_int_list(ks, kernel_size);

            return kernel_size.at(0);
        }

        Value MaxPool2dWithIndicesOpWrapper::getInput() {
            return this->maxpool.self();
        }

        bool MaxPool2dWithIndicesOpWrapper::hasWeights() {
            return false;
        }

        bool MaxPool2dWithIndicesOpWrapper::isDepthWise() {
            return true;
        }

        Operation* MaxPool2dWithIndicesOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                          llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                          llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(!weight.hasValue());
            assert(!bias.hasValue());
            assert(!firstInPartialChain);
            assert(!partialIn.hasValue());

            return builder.create<NPCOMP::aten::MaxPool2dWithIndicesOp>(builder.getUnknownLoc(), returnType, input,
                                                                        this->maxpool.kernel_size(),
                                                                        this->maxpool.stride(),
                                                                        this->maxpool.padding(),
                                                                        this->maxpool.dilation(),
                                                                        this->maxpool.ceil_mode());
        }
    }
}




