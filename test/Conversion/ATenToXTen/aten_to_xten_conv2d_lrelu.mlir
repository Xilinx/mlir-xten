//===- conv2d_xten_affine_ordered.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten -cse | FileCheck %s

// CHECK: "xten.conv2d_lrelu"(%arg0, %0, %1, %2, %2, %2, %int1, %float1.000000e-02) : (!torch.vtensor<[1,4,416,416],f32>, !torch.vtensor<[4,4,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int, !torch.float) -> !torch.vtensor

module attributes {torch.debug_module_name = "CONV2D_NN"}  {
  func @forward(%arg0: !torch.vtensor<[1,4,416,416],f32>) -> !torch.vtensor {
    %int1 = torch.constant.int 1
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %0 = torch.vtensor.literal(dense<"0x93A8A9BDACC802BE00B81FBE480B10BE4B780ABD7330BB3DF089B83DAC9824BEE8EAD43D5C8C05BEA05210BDBE668ABDF67D03BD600906BDD22519BE2E5C13BE8BA5D5BC00410D3B260F9BBD7B94803D279C19BE004938BC6C16163E263A8BBDCF1C1A3EF63A773D1F051ABEF0F5D73D4B8D9DBCA05F98BD6379083E38EA09BECB81923CD6ED0EBD1EC0F33D0B903E3D3303A43D0017713D002A24BD8B910FBDABA40CBE10F3B83D6F81073E7B8CE93DE6C4F0BD2B93F33B169EF0BD0BE6BBBD96E5C6BD464C693DB067CBBD564D6EBB80BFE53B0608DC3DA70201BE5B54EABDFB32C6BD1EE9EE3D106575BD96F40F3E80DB763D00E3AC3C009EA2BA5BF306BDAB4DAB3C20E21BBDAB7E99BA0035AE3C107F05BE4EE7B2BDE6CF093E04810ABE2B1444BC80D4283EC09A763D80A09A3BC4E613BEC816A93D0B20F3BDE0C841BDBB0B65BD4B30CABC56F3EE3A64C40C3E5672AA3C6BDD183E40CCD03DB3CBD7BDE316993D8649F1BD68AA90BDABC9483D4069613DF709003EE0C25BBDABE28B39F8B9AF3DE70C25BE03C7F63DCB200DBEAB8A163B2B21E8BC63C5B3BD2075B43CA2200D3E8B8E47BDC8A5CABD067A7CBDEF2729BE68139A3D16EEA3BD16C9E3BD83A4C4BD6E40C43DDBC2B43D8BFD023E769E773DB0FB63BD1BC53EBD20AE493D3F4B0C3E9B9C393D1380A1BDFE73A0BD2C5C21BE207D133DE08CBFBD8B2918BE52A7153E6BAADDBD8B529BBC6094433D2BFDAD3B4EADE5BD1EF409BE1009083E4054DE3CDA960D3E8BDB64BD4B15493D101C1CBD2BDFF2BB18E3CFBDE6E0663D"> : tensor<4x4x3x3xf32>) : !torch.vtensor<[4,4,3,3],f32>
    %1 = torch.vtensor.literal(dense<[0.0526943207, -0.00343809533, -0.0374836922, -0.102738306]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %5 = torch.aten.conv2d %arg0, %0, %1, %2, %3, %4, %int1 : !torch.vtensor<[1,4,416,416],f32>, !torch.vtensor<[4,4,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
    %6 = torch.aten.leaky_relu %5, %float1.000000e-02 : !torch.vtensor<[?,?,?,?],f32>, !torch.float -> !torch.vtensor
    return %6 : !torch.vtensor
  }
}

