
#ifdef ACAP_OPS
#else
#define ACAP_OPS
#endif

// These ACAP and AIE Ops should be in a different dialect but
// for now they are #included into ATen Dialect from aten.td

def aten_AieLaunchOp : aten_Op<"aie_launch", []>,
                        Arguments<(Variadic<AnyType>:$operands)>,
                        Results<(outs)> {
  let summary = "launch";
  let description = [{
    Launch
  }];
  let regions = (region AnyRegion:$body);
}

def aten_AcapAllocOp: aten_Op<"acap_alloc", []>,
                Results<(outs AnyMemRef)> {
  let arguments = (
    ins Variadic<AnyType>:$events
  );

  let summary = "alloc operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def aten_AcapDeallocOp: aten_Op<"acap_dealloc", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$memref,
        Variadic<AnyType>:$events
  );

  let summary = "dealloc operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def aten_AcapDmaCopyOp: aten_Op<"acap_copy", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$dst,
        AnyMemRef:$src,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def aten_AcapWaitAllOp: aten_Op<"acap_wait_all", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins Variadic<I1>:$events
  );

  let summary = "wait for all operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def aten_AcapDmaLoadOp: aten_Op<"acap_tensor_load", []>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyMemRef:$memref,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def aten_AcapDmaStoreOp: aten_Op<"acap_tensor_store", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$memref,
        AnyTensor:$tensor,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

// 
