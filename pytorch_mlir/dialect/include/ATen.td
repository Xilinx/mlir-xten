// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef ATEN_OPS
#else
#define ATEN_OPS
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"

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
      "std::map<std::string, uint64_t>", "getStatistics"
    >,

    InterfaceMethod<"Return memory transfer requirements of an operand",
      "uint64_t", "getOperandTransferVolume", (ins "unsigned int":$idx, "bool":$read), /*methodBody=*/[{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        if (!read) return 0;
        auto v = *op->getODSOperands(idx).begin();
        auto ty = v.getType();
        return getTensorVolume(ty);
    }]>,

    InterfaceMethod<"Return memory transfer requirements of a result",
      "uint64_t", "getResultTransferVolume", (ins "unsigned int":$idx, "bool":$write), /*methodBody=*/[{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        if (!write) return 0;
        auto v = *op->getODSResults(idx).begin();
        auto ty = v.getType();
        return getTensorVolume(ty);
    }]>,

  ];
}

class aten_Op<string mnemonic, list<OpTrait> traits = [StatisticsOpInterface]> :
    Op<aten_Dialect, mnemonic, traits>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                       "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;

#ifdef ACAP_OPS
#else
include "include/acap.td"
#endif

//#ifdef ACAP_OP_DEFS
//#else
//include "include/aten_op_defs.td"
//#endif

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

def aten_AdaptiveAvgPool2dBackwardOp: aten_Op<"_adaptive_avg_pool2d_backward", [NoSideEffect]>,
                              Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$arg0,
        AnyTensor:$arg1
  );

  let summary = "AdaptiveAvgPool2dBackward operator";
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_AsStridedOp: aten_Op<"as_strided", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
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
    std::map<std::string, uint64_t> getStatistics();
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
    ins AnyTensor:$input,
        AnyTensor:$weight,
        AnyTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation,
        AnyScalar:$transposed,
        AnyType:$output_padding,
        AnyScalar:$groups,
        AnyType:$arg9,
        AnyType:$arg10,
        AnyType:$arg11
  );

  let summary = "Convolution operator";
  let description = [{
    Convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    //uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    //uint64_t getResultTransferVolume(unsigned int idx, bool read);
	}];
}

def aten_ConvolutionBackwardOp: aten_Op<"_convolution_backward", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
	}];
}

def aten_DivOp: aten_Op<"div", [NoSideEffect, StatisticsOpInterface]>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b
  );

  let summary = "div operator";
  let description = [{
    div operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_DivUnderOp: aten_Op<"div_", [NoSideEffect, StatisticsOpInterface]>,
                     Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b
  );

  let summary = "In-place div operator";
  let description = [{
    In-place div operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
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

def aten_ExpandOp: aten_Op<"expand", [NoSideEffect, StatisticsOpInterface]>,
                       Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2
  );

  let summary = "expand operator";
  let description = [{
    expand operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_FlattenOp: aten_Op<"flatten", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_GatherOp: aten_Op<"gather", [NoSideEffect]>,
                       Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3
  );

  let summary = "gather operator";
  let description = [{
    gather operator
  }];
}

def aten_HardtanhOp: aten_Op<"hardtanh", [NoSideEffect, StatisticsOpInterface]>,
                          Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$arg0,
        AnyScalar:$arg1,
        AnyScalar:$arg2
  );

  let summary = "hardtanh operator";
  let description = [{
    hardtanh operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_HardtanhUnderOp: aten_Op<"hardtanh_", [NoSideEffect, StatisticsOpInterface]>,
                          Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$arg0,
        AnyScalar:$arg1,
        AnyScalar:$arg2
  );

  let summary = "hardtanh operator";
  let description = [{
    hardtanh operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_HardtanhBackwardOp: aten_Op<"hardtanh_backward", [NoSideEffect/*, StatisticsOpInterface*/]>,
                             Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$arg0,
        AnyTensor:$arg1,
        AnyScalar:$arg2,
        AnyScalar:$arg3
  );

  let summary = "hardtanh backward operator";
  let description = [{
    hardtanh operator
  }];
  //let extraClassDeclaration = [{
  //  std::map<std::string, uint64_t> getStatistics();
  //}];
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MaxPool2dWithIndicesBackwardOp: aten_Op<"max_pool2d_with_indices_backward", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];

}

def aten_MeanOp: aten_Op<"mean", [NoSideEffect, StatisticsOpInterface]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "mean operator";
  let description = [{
    mean operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MMOp: aten_Op<"mm", [NoSideEffect, SameOperandsAndResultElementType, StatisticsOpInterface]>,
               Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyTensor:$y
  );

  let summary = "matrix multiply operator";
  let description = [{
    matrix multiply operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NativeBatchNormBackwardOp: aten_Op<"native_batch_norm_backward", [NoSideEffect, StatisticsOpInterface]>,
                            Results<(outs AnyTensor:$dx, AnyTensor:$dm, AnyTensor:$dv)> {
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

  let summary = "BatchNormBackward operator";
  let description = [{
    BatchNormBackward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_NegOp: aten_Op<"neg", [NoSideEffect, SameOperandsAndResultElementType]>,
               Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "neg operator";
  let description = [{
    neg operator
  }];
}

def aten_NllLoss2dBackwardOp: aten_Op<"nll_loss2d_backward", [NoSideEffect]>,
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

  let summary = "NllLoss2dBackward operator";
  let description = [{
    NllLoss2dBackward operator
  }];
}

def aten_NllLoss2dForwardOp: aten_Op<"nll_loss2d_forward", [NoSideEffect]>,
                             Results<(outs AnyTensor, AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4
  );

  let summary = "NllLoss2dForward operator";
  let description = [{
    NllLoss2dForward operator
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

def aten_SizeOp : aten_Op<"size", [NoSideEffect]>,
                  Results<(outs AnyScalar)> {
let arguments = (
    ins AnyTensor:$a,
        AnyScalar:$dim
  );

  let summary = "size operator";
  let description = [{
    size operator
  }];
}

def aten_SubOp: aten_Op<"sub", [NoSideEffect, StatisticsOpInterface]>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b,
        AnyScalar:$c
  );

  let summary = "sub operator";
  let description = [{
    sub operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SubUnderOp: aten_Op<"sub_", [NoSideEffect, StatisticsOpInterface]>,
                     Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$a,
        AnyScalarOrTensor:$b,
        AnyScalar:$c
  );

  let summary = "In-place sub operator";
  let description = [{
    In-place sub operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SumOp: aten_Op<"sum", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_SqueezeOp: aten_Op<"squeeze", [NoSideEffect]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$dim
  );

  let summary = "Squeeze operator";
  let description = [{
    Squeeze operator
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
    std::map<std::string, uint64_t> getStatistics();
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
    std::map<std::string, uint64_t> getStatistics();
	}];
}

def aten_TransposeOp: aten_Op<"t", [NoSideEffect, SameOperandsAndResultElementType, StatisticsOpInterface]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "Transpose operator";
  let description = [{
    Transpose operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
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

def aten_ThresholdBackwardOp: aten_Op<"threshold_backward", [NoSideEffect, StatisticsOpInterface]>,
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
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_TypeCastOp : aten_Op<"type_cast", [NoSideEffect]>,
                      Results<(outs AnyType)> {
  let summary = "TypeCast operator";
  let arguments = (
    ins AnyType:$x
  );
}

def aten_UnsqueezeOp: aten_Op<"unsqueeze", [NoSideEffect, StatisticsOpInterface]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$dim
  );

  let summary = "Unsqueeze operator";
  let description = [{
    Unsqueeze operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_ViewOp: aten_Op<"view", [NoSideEffect, StatisticsOpInterface]>,
                 Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x,
        AnyType:$size
  );

  let summary = "View operator";
  let description = [{
    View operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}
