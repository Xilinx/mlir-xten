# PyTorch/ACAP

PyTorch/ACAP is a prototype PyTorch backend targeting Xilinx ACAP devices.
It is designed to be a just-in-time compiler (JIT) front-end as well as a PyTorch to MLIR exporter.
PyTorch/ACAP closely follows the design of [PyTorch/XLA](http://github.com/pytorch/xla).

PyTorch/ACAP currently implements:
 - A PyTorch ACAP device type and JIT front-end.
 - An MLIR dialect for expressing PyTorch ATen operations.
 - MLIR passes for ATen dialect analysis and lowering.
 
The PyTorch ACAP device allows existing PyTorch scripts to target an ACAP device as easily as a GPU or CPU device.  The MLIR dialect export functionality allows MLIR and LLVM based compilers to be used as the optimizer and code generator for the PyTorch/ACAP JIT.

## MLIR Export Example
Here we demonstrate the MLIR generation and export capability of the PyTorch/ACAP.  The following is a script containing a very simple PyTorch Model, it just adds two tensors.


```python
import torch
import torch.nn as nn
import torch_acap

class adder(nn.Module):
        def __init__(self):
            super(adder, self).__init__()
        def forward(self, in0, in1):
            return in0 + in1

cpu_model = adder()
```

We can run the model and get a result:


```python
cpu_tensor = torch.randn(1,2,3)
cpu_result = cpu_model(cpu_tensor, cpu_tensor)
print(cpu_result)
```

    tensor([[[-0.5024, -0.1082, -0.4396],
             [-1.5719,  0.1685, -0.6653]]])


We can also move the model and data to the ACAP device just like we would when using a GPU:


```python
# get the default ACAP device
dev = torch_acap.acap_device()

# move the data to the acap device
acap_tensor = cpu_tensor.to(dev)
acap_model = adder().to(dev)
```

We can now run the model as before.  But instead of printing the result tensor, we'll ask the system for the MLIR corresponding to the computation:


```python
acap_result = acap_model(acap_tensor, acap_tensor)

mlir = torch_acap.get_mlir( acap_result )

print(mlir)
```

    
    
    module {
      func @graph(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
        %0 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
        %1 = "aten.add"(%arg0, %arg1, %0) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, i32) -> tensor<1x2x3xf32>
        return %1 : tensor<1x2x3xf32>
      }
    }
    


The ACAP device implements a lazy JIT compiler front-end.  That is, it will continue adding to the MLIR until forced to compile.

For example, we can continue adding to the above compute graph...


```python
# a second add operation...
acap_result = acap_result + torch.randn(1,2,3,device=dev)

# ...and a mul operation
acap_result = acap_result * torch.randn(1,2,3,device=dev)

mlir = torch_acap.get_mlir( acap_result )
print(mlir)
```

    
    
    module {
      func @graph(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2x3xf32>, %arg2: tensor<1x2x3xf32>, %arg3: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
        %0 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
        %1 = "aten.add"(%arg0, %arg1, %0) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, i32) -> tensor<1x2x3xf32>
        %2 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
        %3 = "aten.add"(%1, %arg2, %2) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, i32) -> tensor<1x2x3xf32>
        %4 = "aten.mul"(%3, %arg3) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
        return %4 : tensor<1x2x3xf32>
      }
    }
    


## Pytorch/ACAP JIT Example


```python
# TODO
```
