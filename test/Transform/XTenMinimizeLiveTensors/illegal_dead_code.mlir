// RUN: not aten-opt %s -xten-minimize-live -split-input-file -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK_SCHEDULE_ERROR

// Illegal dead code where (dead) code use some xten operands.
// CHECK-LABEL: illegal_dead_code
// CHECK_SCHEDULE_ERROR: function cannot be rescheduled due to illegal dead code, aborting
func.func @illegal_dead_code(%arg0: !torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %1 = torch.vtensor.literal(dense<0.00999999977> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %2 = torch.vtensor.literal(dense<2.000000e-02> : tensor<64x64x1x1xf32>) : !torch.vtensor<[64,64,1,1],f32>
  %3 = torch.vtensor.literal(dense<2.000000e-02> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %4 = torch.vtensor.literal(dense<2.000000e-02> : tensor<256x64x1x1xf32>) : !torch.vtensor<[256,64,1,1],f32>
  %5 = torch.vtensor.literal(dense<0.00999999977> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %29 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %32 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>

  %33 = "xten.conv2d"(%arg0, %4, %5, %29, %32, %29, %int1) {layer_name = "conv2d0"} : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[256,64,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,56,56],f32>
  %34 = "xten.conv2d"(%arg0, %4, %5, %29, %32, %29, %int1) {layer_name = "conv2d1"} : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[256,64,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,56,56],f32>
  %35 = "xten.conv2d_tensoradd_relu"(%arg0, %4, %5, %29, %32, %29, %int1, %33) {layer_name = "conv2d_tensoradd_relu0"} : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[256,64,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32>
  %36 = "xten.add"(%33, %34) {layer_name = "add1"} : (!torch.vtensor<[1,256,56,56],f32>, !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32>
  return %36 : !torch.vtensor<[1,256,56,56],f32>
}

// -----

// Illegal dead code where (dead) code use some xten operands.
// CHECK-LABEL: illegal_dead_code
// CHECK_SCHEDULE_ERROR: function cannot be rescheduled due to illegal dead code, aborting
func.func @illegal_dead_code(%arg0: !torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %1 = torch.vtensor.literal(dense<0.00999999977> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %2 = torch.vtensor.literal(dense<2.000000e-02> : tensor<64x64x1x1xf32>) : !torch.vtensor<[64,64,1,1],f32>
  %3 = torch.vtensor.literal(dense<2.000000e-02> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %4 = torch.vtensor.literal(dense<2.000000e-02> : tensor<256x64x1x1xf32>) : !torch.vtensor<[256,64,1,1],f32>
  %5 = torch.vtensor.literal(dense<0.00999999977> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %29 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %32 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>

  %33 = "xten.conv2d"(%arg0, %4, %5, %29, %32, %29, %int1) {layer_name = "conv2d0"} : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[256,64,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,56,56],f32>
  %34 = "xten.conv2d"(%arg0, %4, %5, %29, %32, %29, %int1) {layer_name = "conv2d1"} : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[256,64,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,56,56],f32>
  %35 = "xten.add"(%33, %34) {layer_name = "add0"} : (!torch.vtensor<[1,256,56,56],f32>, !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32>
  %36 = "xten.add"(%35, %35) {layer_name = "add1"} : (!torch.vtensor<[1,256,56,56],f32>, !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32>
  %37 = "xten.add"(%33, %34) {layer_name = "add2"} : (!torch.vtensor<[1,256,56,56],f32>, !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,56,56],f32>
  return %37 : !torch.vtensor<[1,256,56,56],f32>
}