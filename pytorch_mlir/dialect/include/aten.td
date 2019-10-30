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

class aten_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<aten_Dialect, mnemonic, traits>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnyInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                        "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[AnyInteger.predicate,
                                   AnyFloat.predicate]>,
                                "scalar">;

def aten_TypeCastOp : aten_Op<"type_cast", [NoSideEffect]>,
                              Results<(outs AnyType)> {
  let summary = "TypeCast operator";
  let arguments = (
    ins AnyType:$x
  );
}

def aten_AllocOp : aten_Op<"alloc", [NoSideEffect]>,
                           Results<(outs AnyType)> {
  let summary = "Alloc operator";
}

def aten_AddOp: aten_Op<"add", [NoSideEffect]>,
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
}

def aten_ConstantOp: aten_Op<"constant", [NoSideEffect]>,
                        Results<(outs AnyType)> {
  let summary = "Constant operator";
  let description = [{
    Constant operator
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

def aten_ReLUOp: aten_Op<"relu", [NoSideEffect]>,
                        Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$x
  );

  let summary = "ReLU operator";
  let description = [{
    ReLU operator
  }];

}

def aten_BatchNormOp: aten_Op<"batch_norm", [NoSideEffect]>,
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
        AnyType:$arg8
  );

  let summary = "BatchNorm operator";
  let description = [{
    BatchNorm operator
  }];

}

def aten_ConvolutionOp: aten_Op<"_convolution", [NoSideEffect]>,
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

def aten_AddmmOp: aten_Op<"addmm", [NoSideEffect]>,
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

}

def aten_MaxPool2dOp: aten_Op<"max_pool2d", [NoSideEffect]>,
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

}

def aten_MaxPool2dWithIndiciesOp: aten_Op<"max_pool2d_with_indices", [NoSideEffect]>,
                        Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5
  );

  let summary = "MaxPool2dWithIndicies operator";
  let description = [{
    MaxPool2dWithIndicies operator
  }];

}
