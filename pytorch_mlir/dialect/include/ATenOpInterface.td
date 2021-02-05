// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
include "mlir/IR/OpBase.td"

#ifdef ATEN_OP_INTERFACE
#else
#define ATEN_OP_INTERFACE
#endif

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
        return xilinx::air::getTensorVolume(ty);
    }]>,

    InterfaceMethod<"Return memory transfer requirements of a result",
      "uint64_t", "getResultTransferVolume", (ins "unsigned int":$idx, "bool":$write), /*methodBody=*/[{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        if (!write) return 0;
        auto v = *op->getODSResults(idx).begin();
        auto ty = v.getType();
        return xilinx::air::getTensorVolume(ty);
    }]>,

  ];
}
