//===- EnclaveInterface.td - XTenNN EnclaveOp interface ----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the XTenNN EnclaveOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_ENCLAVEOPINTERFACE
#define XTENNN_ENCLAVEOPINTERFACE

include "mlir/IR/OpBase.td"

def XTenNN_EnclaveOp : OpInterface<"EnclaveOp"> {
    let cppNamespace = "amd::xten_nn";

    let description = [{
        Interface for an operation that forms an isolated enclave around some
        captured values.

        It is required that all values passed to an appropriate region
        terminator will be returned from the enclave in the same order.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the range of captured values.}],
            /*retTy=*/"::mlir::OperandRange",
            /*methodName=*/"getCaptures",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return $_op.getOperands(); }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the enclave body block, if present.}],
            /*retTy=*/"::mlir::Block *",
            /*methodName=*/"getOptionalEnclaveBody",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto &region = $_op->getRegion(0);
                if (region.hasOneBlock())
                    return &region.front();
                return {};
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the enclave body block.}],
            /*retTy=*/"::mlir::Block &",
            /*methodName=*/"getEnclaveBody",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto &region = $_op->getRegion(0);
                assert(region.hasOneBlock());
                return region.front();
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the BlockArgument that provides a captured value.}],
            /*retTy=*/"::mlir::BlockArgument",
            /*methodName=*/"lookupArgument",
            /*args=*/(ins "::mlir::OpOperand &":$capture),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                assert(capture.getOwner() == $_op);
                return getEnclaveBody().getArgument(capture.getOperandNumber());
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Populates a map from capture values to block arguments.}],
            /*retTy=*/"void",
            /*methodName=*/"populateCaptureMap",
            /*args=*/(ins "::mlir::IRMapping &":$result),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                enclave_interface_defaults::populateCaptureMap($_op, result);
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Captures an additional SSA value.}],
            /*retTy=*/"void",
            /*methodName=*/"capture",
            /*args=*/(ins
                "::mlir::ValueRange":$values,
                "::mlir::IRMapping &":$result),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                enclave_interface_defaults::capture($_op, values, result);
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{Removes an unused SSA value from the captures.}],
            /*retTy=*/"void",
            /*methodName=*/"uncapture",
            /*args=*/(ins "::llvm::ArrayRef<::mlir::BlockArgument>":$args),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                enclave_interface_defaults::uncapture($_op, args);
            }]
        >
    ];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::IsIsolatedFromAbove>(),
            "`EnclaveInterface` is only applicable to `IsolatedFromAbove` ops"
        );

        return enclave_interface_defaults::verify($_op);
    }];

    code extraSharedClassDeclaration = [{
        /// Gets the number of captured SSA values.
        unsigned getNumCaptures() { return $_op.getCaptures().size(); }

        /// Gets the captured SSA value at the specified index.
        ::mlir::Value getCapture(unsigned index)
        {
            return $_op.getCaptures()[index];
        }

        /// Gets the terminator of the body block.
        ///
        /// @pre        getEnclaveBody() has a terminator
        ::mlir::Operation* getTerminator()
        {
            return $_op.getEnclaveBody().getTerminator();
        }
    }];
}

#endif

