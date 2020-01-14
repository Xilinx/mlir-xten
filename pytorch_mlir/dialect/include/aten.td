// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef ATEN_OPS
#else
#define ATEN_OPS
#endif

def aten_Dialect : Dialect {
  let name = "aten";
}

def StatisticsOpInterface : OpInterface<"StatisticsOpInterface"> {
  let description = [{
     This interface allows ops to expose a static operation profile,
	  describing the computational behavior of their function.
  }];

  let methods = [
    InterfaceMethod<
      "Return statistics about the compute requirements of an op",
      "std::map<std::string, unsigned>", "updateStatistics"
    >,
  ];
}

class aten_Op<string mnemonic, list<OpTrait> traits = [StatisticsOpInterface]> :
    Op<aten_Dialect, mnemonic, traits>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnyInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                       "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[AnyInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;


def aten_AdaptiveAvgPool2dOp: aten_Op<"_adaptive_avg_pool2d", [NoSideEffect]>,
                              Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1
  );

  let summary = "AdaptiveAvgPool2d operator";
  let description = [{
    AdaptiveAvgPool2d operator
  }];
}

def aten_AddOp: aten_Op<"add", [NoSideEffect, StatisticsOpInterface]>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b,
        AnyScalar:$c
  );

  let summary = "Add operator";
  let description = [{
    Add operator
  }];
  let extraClassDeclaration = [{
    /// Return the statistics.
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;

      Type resultTy = getResult()->getType();
      unsigned ofm_volume = getTensorVolume(resultTy);
      toReturn["+"] = ofm_volume;
      toReturn["activation_out"] = ofm_volume;

      TensorType tensorResultTy = resultTy.cast<TensorType>();
      unsigned num_output_neurons = tensorResultTy.getShape()[1];

      // Find the size of the A and B operands
      Type aType = getOperand(0)->getType();
      unsigned a_volume = getTensorVolume(aType);

      Type bType = getOperand(1)->getType();
      unsigned b_volume = getTensorVolume(bType);

      toReturn["activation_in"] = a_volume + b_volume;
      return(toReturn);
    }
  }];
}

def aten_AddUnderOp: aten_Op<"add_", [NoSideEffect, StatisticsOpInterface]>,
                     Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b,
        AnyScalar:$c
  );

  let summary = "In-place add operator";
  let description = [{
    In-place add operator
  }];
  let extraClassDeclaration = [{
    /// Return the statistics.
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;

      Type resultTy = getResult()->getType();
      unsigned ofm_volume = getTensorVolume(resultTy);
      toReturn["+"] = ofm_volume;
      toReturn["activation_out"] = ofm_volume;

      TensorType tensorResultTy = resultTy.cast<TensorType>();
      unsigned num_output_neurons = tensorResultTy.getShape()[1];

      // Find the size of the A and B operands
      Type aType = getOperand(0)->getType();
      unsigned a_volume = getTensorVolume(aType);

      Type bType = getOperand(1)->getType();
      unsigned b_volume = getTensorVolume(bType);

      toReturn["activation_in"] = a_volume + b_volume;
      return(toReturn);
    }
  }];
}

def aten_AddmmOp: aten_Op<"addmm", [NoSideEffect, StatisticsOpInterface]>,
                  Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4
  );

  let summary = "Addmm operator";
  let description = [{
    Addmm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {

      std::map<std::string, unsigned> toReturn;
      // For linear, we need the number of output neurons and the number of input neurons
      // Then the number of forward MACs is input * output
      // And the number of adds is output if there is bias

      Type resultTy = getResult()->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();
      unsigned num_output_neurons = tensorResultTy.getShape()[1];
      unsigned ofm_volume = getTensorVolume(tensorResultTy);

      // Use the weight tensor to find the number of input neurons
      Type wType = getOperand(2)->getType();
      TensorType wTy = wType.cast<TensorType>();
      unsigned num_input_neurons = wTy.getShape()[0];
      unsigned total_MACs = ofm_volume * num_input_neurons;
      unsigned weight_volume = getTensorVolume(wTy);

      Type ifmType = getOperand(1)->getType();
      TensorType txTy = ifmType.cast<TensorType>();
      unsigned ifm_volume = getTensorVolume(txTy);

      toReturn["MAC"] = total_MACs;
      toReturn["+"] = ofm_volume;   // Should be gated on whether there is bias at all
      toReturn["activation_in"] = ifm_volume;
      toReturn["activation_out"] = ofm_volume;
      toReturn["parameters_in"] = weight_volume + num_output_neurons;

      return(toReturn);
    }
  }];
}

def aten_AsStridedOp: aten_Op<"as_strided", [NoSideEffect]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$size,
        AnyType:$stride
  );

  let summary = "as_strided operator";
  let description = [{
    as_strided operator
  }];
}

def aten_AllocOp : aten_Op<"alloc", [NoSideEffect]>,
                   Results<(outs AnyType)> {
  let summary = "Alloc operator";
}

def aten_BatchNormOp: aten_Op<"batch_norm", [NoSideEffect, StatisticsOpInterface]>,
                      Results<(outs AnyTensor:$output, AnyTensor:$save_mean, AnyTensor:$save_invstd)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7,
        AnyType:$arg8
  );

  let summary = "BatchNorm operator";
  let description = [{
    BatchNorm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;
      Type resultTy = getResult(0)->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned op_volume = getTensorVolume(tensorResultTy);
      toReturn["activation_in"] = op_volume;
      toReturn["activation_out"] = op_volume;

      // There are 2x as many parameters are there are planes ...
      unsigned ifm_depth = tensorResultTy.getShape()[1];
      toReturn["parameters_in"] = ifm_depth * 2;

      // Now for the arithmetic.  Assume variance is calculated as sum of squares

      toReturn["+"] = op_volume;   // Add up for mean
      toReturn["*"] = op_volume;   // Square for variance
      toReturn["+"] += op_volume;  // Add up squares for variance

      toReturn["*"] += ifm_depth;   // Calc channel means
      toReturn["-"] += ifm_depth;   // Calc channel vars
      toReturn["*"] += ifm_depth;   // Calc channel vars

      toReturn["sqrt"] = ifm_depth;  // Convert to SD
      toReturn["/"] += ifm_depth;    // Get the reciprocal

      toReturn["+"] += op_volume;   // Subtract mean off each pixel
      toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

      toReturn["+"] += op_volume;   // Bias
      toReturn["*"] += op_volume;   // Scale

      return(toReturn);
    }
  }];
}

def aten_ConstantOp: aten_Op<"constant", [NoSideEffect]>,
                     Results<(outs AnyType)> {
  let summary = "Constant operator";
  let description = [{
    Constant operator
  }];

}

def aten_ConvolutionOp: aten_Op<"_convolution", [NoSideEffect, StatisticsOpInterface]>,
                        Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7,
        AnyType:$arg8,
        AnyType:$arg9,
        AnyType:$arg10,
        AnyType:$arg11
  );

  let summary = "Convolution operator";
  let description = [{
    Convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {

      // XXX why dont I see stride in here?

      std::map<std::string, unsigned> toReturn;
      // For convolution, we need the OFM volume.
      // Then the number of forward MACs per pixel are kernel_width * kernel_height * ifm_depth / groups

      Type resultTy = getResult()->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned ofm_volume = getTensorVolume(tensorResultTy);
      unsigned ofm_depth = tensorResultTy.getShape()[1];
      // using namespace edsc;
      // using call = intrinsics::ValueBuilder<mlir::CallOp>;
      // using constInt = intrinsics::constant_int;

      // All the info we need for MACs is in the weight tensor
      Type wType = getOperand(1)->getType();
      TensorType wTy = wType.cast<TensorType>();

      unsigned ifm_depth = wTy.getShape()[1];
      unsigned kernel_width = wTy.getShape()[2];
      unsigned kernel_height = wTy.getShape()[3];
      unsigned groups = 1; // It's one of the operands, not sure which

      unsigned MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
      unsigned total_MACs = ofm_volume * MACs_per_OFM;

      Type ifmType = getOperand(0)->getType();
      TensorType txTy = ifmType.cast<TensorType>();
      unsigned ifm_volume = getTensorVolume(txTy);
      unsigned weight_volume = getTensorVolume(wTy);

      unsigned bias_volume = ofm_depth;  // See below
      toReturn["+"] = ofm_volume;        // Should be gated on whether there is bias at all

      toReturn["MAC"] = total_MACs;
      toReturn["activation_in"] = ifm_volume;
      toReturn["activation_out"] = ofm_volume;
      toReturn["parameters_in"] = weight_volume + ofm_depth;

      return(toReturn);
    }
	}];
}

def aten_ConvolutionBackwardOp: aten_Op<"_convolution_backward", [NoSideEffect]>,
                                Results<(outs AnyTensor:$dx, AnyTensor:$dw, AnyTensor:$db)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7,
        AnyType:$arg8
  );

  let summary = "ConvolutionBackward operator";
  let description = [{
    ConvolutionBackward operator
  }];
}

def aten_DropoutOp: aten_Op<"dropout", [NoSideEffect]>,
                    Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2
  );

  let summary = "Dropout operator";
  let description = [{
    Dropout operator
  }];
}

def aten_FlattenOp: aten_Op<"flatten", [NoSideEffect]>,
                    Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2
  );

  let summary = "Flatten operator";
  let description = [{
    Flatten operator
  }];
}

def aten_LogSoftmaxOp: aten_Op<"_log_softmax", [NoSideEffect]>,
                       Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2
  );

  let summary = "LogSoftmax operator";
  let description = [{
    LogSoftmax operator
  }];
}

def aten_LogSoftmaxBackwardOp: aten_Op<"_log_softmax_backward_data", [NoSideEffect]>,
                               Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3
  );

  let summary = "LogSoftmaxBackward operator";
  let description = [{
    LogSoftmaxBackward operator
  }];
}

def aten_MaxPool2dOp: aten_Op<"max_pool2d", [NoSideEffect, StatisticsOpInterface]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5
  );

  let summary = "MaxPool2d operator";
  let description = [{
    MaxPool2d operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {

      auto unpack = [](auto &op, auto &v) -> void {
                      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                      for (auto i : a.getIntValues())
                        v.push_back(i.getSExtValue());
                    };

      std::map<std::string, unsigned>  toReturn;
      Type resultTy = getResult()->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned ofm_volume = getTensorVolume(tensorResultTy);
      toReturn["activation_out"] = ofm_volume;

      Type ifmType = getOperand(0)->getType();
      TensorType txTy = ifmType.cast<TensorType>();
      unsigned ifm_volume = getTensorVolume(txTy);
      toReturn["activation_in"] = ifm_volume;

      // To find the number of compares, we need the filter extent

      std::vector<uint64_t> kernel;
      mlir::Value *k = getOperand(1);
      unpack(k, kernel);

      unsigned aperture = kernel[0] * kernel[1];
      toReturn[">"] = ofm_volume * aperture;

      return(toReturn);
    }
  }];
}

def aten_MaxPool2dWithIndicesOp: aten_Op<"max_pool2d_with_indices", [NoSideEffect, StatisticsOpInterface]>,
                                 Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5
  );

  let summary = "MaxPool2dWithIndices operator";
  let description = [{
    MaxPool2dWithIndices operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {

      auto unpack = [](auto &op, auto &v) -> void {
                      auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                      DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                      for (auto i : a.getIntValues())
                        v.push_back(i.getSExtValue());
                    };

      std::map<std::string, unsigned>  toReturn;
      Type resultTy = getResult(0)->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned ofm_volume = getTensorVolume(tensorResultTy);
      toReturn["activation_out"] = ofm_volume;

      Type ifmType = getOperand(0)->getType();
      TensorType txTy = ifmType.cast<TensorType>();
      unsigned ifm_volume = getTensorVolume(txTy);
      toReturn["activation_in"] = ifm_volume;

      // To find the number of compares, we need the filter extent

      std::vector<uint64_t> kernel;
      mlir::Value *k = getOperand(1);
      unpack(k, kernel);

      unsigned aperture = kernel[0] * kernel[1];
      toReturn[">"] = ofm_volume * aperture;

      return(toReturn);
    }
  }];
}

def aten_MaxPool2dWithIndicesBackwardOp: aten_Op<"max_pool2d_with_indices_backward", [NoSideEffect]>,
                                         Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7
  );

  let summary = "MaxPool2dWithIndicesBackward operator";
  let description = [{
    MaxPool2dWithIndicesBackward operator
  }];
}

def aten_MeanOp: aten_Op<"mean", [NoSideEffect]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$dim,
        AnyType:$keepdim
  );

  let summary = "mean operator";
  let description = [{
    mean operator
  }];
}

def aten_MMOp: aten_Op<"mm", [NoSideEffect, SameOperandsAndResultElementType]>,
               Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyTensor:$y
  );

  let summary = "matrix multiply operator";
  let description = [{
    matrix multiply operator
  }];
}

def aten_MulOp: aten_Op<"mul", [NoSideEffect, StatisticsOpInterface]>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b
  );

  let summary = "mul operator";
  let description = [{
    mul operator
  }];
  let extraClassDeclaration = [{
    /// Return the statistics.
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;

      Type resultTy = getResult()->getType();
      unsigned ofm_volume = getTensorVolume(resultTy);
      toReturn["+"] = ofm_volume;
      toReturn["activation_out"] = ofm_volume;

      TensorType tensorResultTy = resultTy.cast<TensorType>();
      unsigned num_output_neurons = tensorResultTy.getShape()[1];

      // Find the size of the A and B operands
      Type aType = getOperand(0)->getType();
      unsigned a_volume = getTensorVolume(aType);

      Type bType = getOperand(1)->getType();
      unsigned b_volume = getTensorVolume(bType);

      toReturn["activation_in"] = a_volume + b_volume;
      return(toReturn);
    }
  }];
}

def aten_MulUnderOp: aten_Op<"mul_", [NoSideEffect, StatisticsOpInterface]>,
                     Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b
  );

  let summary = "In-place mul operator";
  let description = [{
    In-place mul operator
  }];
  let extraClassDeclaration = [{
    /// Return the statistics.
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;

      Type resultTy = getResult()->getType();
      unsigned ofm_volume = getTensorVolume(resultTy);
      toReturn["+"] = ofm_volume;
      toReturn["activation_out"] = ofm_volume;

      TensorType tensorResultTy = resultTy.cast<TensorType>();
      unsigned num_output_neurons = tensorResultTy.getShape()[1];

      // Find the size of the A and B operands
      Type aType = getOperand(0)->getType();
      unsigned a_volume = getTensorVolume(aType);

      Type bType = getOperand(1)->getType();
      unsigned b_volume = getTensorVolume(bType);

      toReturn["activation_in"] = a_volume + b_volume;
      return(toReturn);
    }
  }];
}

def aten_NativeBatchNormOp: aten_Op<"native_batch_norm", [NoSideEffect, StatisticsOpInterface]>,
                            Results<(outs AnyTensor:$output, AnyTensor:$save_mean, AnyTensor:$save_invstd)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7
  );

  let summary = "BatchNorm operator";
  let description = [{
    BatchNorm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {
            std::map<std::string, unsigned>  toReturn;
            Type resultTy = getResult(0)->getType();
            TensorType tensorResultTy = resultTy.cast<TensorType>();

            unsigned op_volume = getTensorVolume(tensorResultTy);
            toReturn["activation_in"] = op_volume;
            toReturn["activation_out"] = op_volume;

            // There are 2x as many parameters are there are planes ...
            unsigned ifm_depth = tensorResultTy.getShape()[1];
            toReturn["parameters_in"] = ifm_depth * 2;

            // Now for the arithmetic.  Assume variance is calculated as sum of squares

            toReturn["+"] = op_volume;   // Add up for mean
            toReturn["*"] = op_volume;   // Square for variance
            toReturn["+"] += op_volume;  // Add up squares for variance

            toReturn["*"] += ifm_depth;   // Calc channel means
            toReturn["-"] += ifm_depth;   // Calc channel vars
            toReturn["*"] += ifm_depth;   // Calc channel vars

            toReturn["sqrt"] = ifm_depth;  // Convert to SD
            toReturn["/"] += ifm_depth;    // Get the reciprocal

            toReturn["+"] += op_volume;   // Subtract mean off each pixel
            toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

            toReturn["+"] += op_volume;   // Bias
            toReturn["*"] += op_volume;   // Scale


            return(toReturn);
        }

  }];
}

def aten_NllLossBackwardOp: aten_Op<"nll_loss_backward", [NoSideEffect]>,
                           Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6
  );

  let summary = "NllLossBackward operator";
  let description = [{
    NllLossBackward operator
  }];
}

def aten_NllLossForwardOp: aten_Op<"nll_loss_forward", [NoSideEffect]>,
                           Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4
  );

  let summary = "NllLossForward operator";
  let description = [{
    NllLossForward operator
  }];
}

def aten_SumOp: aten_Op<"sum", [NoSideEffect]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$dim,
        AnyType:$keepdim
  );

  let summary = "sum operator";
  let description = [{
    sum operator
  }];
}

def aten_ReLUOp: aten_Op<"relu", [NoSideEffect, StatisticsOpInterface]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "ReLU operator";
  let description = [{
    ReLU operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;
      Type resultTy = getResult()->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned op_volume = getTensorVolume(tensorResultTy);
      toReturn["activation_in"] = op_volume;
      toReturn["activation_out"] = op_volume;
      toReturn[">"] = op_volume;

      return(toReturn);
    }
	}];
}

def aten_ReLUUnderOp: aten_Op<"relu_", [NoSideEffect, StatisticsOpInterface]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "In-Place ReLU operator";
  let description = [{
    In-Place ReLU operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, unsigned>  updateStatistics() {
      std::map<std::string, unsigned> toReturn;
      Type resultTy = getResult()->getType();
      TensorType tensorResultTy = resultTy.cast<TensorType>();

      unsigned op_volume = getTensorVolume(tensorResultTy);
      toReturn["activation_in"] = op_volume;
      toReturn["activation_out"] = op_volume;
      toReturn[">"] = op_volume;

      return(toReturn);
    }
	}];
}

def aten_ThresholdBackwardOp: aten_Op<"threshold_backward", [NoSideEffect]>,
                              Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$arg0,
        AnyTensor:$arg1,
        AnyType:$arg2
  );

  let summary = "ThresholdBackward operator";
  let description = [{
    ThresholdBackward operator
  }];

}

def aten_TransposeOp: aten_Op<"t", [NoSideEffect, SameOperandsAndResultElementType]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "Transpose operator";
  let description = [{
    Transpose operator
  }];

}

def aten_ThresholdOp: aten_Op<"threshold", [NoSideEffect]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyFloat:$arg1,
        AnyFloat:$arg2
  );

  let summary = "Threshold operator";
  let description = [{
    Threshold operator
  }];

}

def aten_TypeCastOp : aten_Op<"type_cast", [NoSideEffect]>,
                      Results<(outs AnyType)> {
  let summary = "TypeCast operator";
  let arguments = (
    ins AnyType:$x
  );
}

def aten_ViewOp: aten_Op<"view", [NoSideEffect]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$size
  );

  let summary = "View operator";
  let description = [{
    View operator
  }];
}
