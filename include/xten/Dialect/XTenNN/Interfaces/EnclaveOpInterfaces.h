//===- EnclaveOpInterfaces.h -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//
//===----------------------------------------------------------------------===//
//
// Declares the XTenNN EnclaveOp interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"

namespace amd::xten_nn {

namespace enclave_interface_defaults {

/// Populates @p result with mappings from captured values to block arguments.
///
/// @pre        `isa<EnclaveOp>(op)`
void populateCaptureMap(mlir::Operation *op,
                        mlir::IRMapping &result);

/// Ensures that @p values are captured by @p op and return their corresponding
/// block arguments.
///
/// This default implementation assumes that operands are mapped 1-1 to block
/// arguments. It will therefore append operands and arguments to the end of
/// their respective lists if new captures need to be added.
///
/// @pre        `isa<EnclaveOp>(op)`
/// @pre        @p values are defined outside of @p op
/// @post       `result.contains(value)`
void capture(mlir::Operation *op, mlir::ValueRange values,
             mlir::IRMapping &result);

/// Removes the dead captures corresponding to @p args .
///
/// This default implementation assumes that operands are mapped 1-1 to block
/// arguments. It will therefore remove operands and arguments at the same
/// indices.
///
/// @pre        `isa<EnclaveOp>(op)`
/// @pre        @p args have no remaining uses
/// @pre        @p args are defined inside @p op
void uncapture(mlir::Operation *op, mlir::ArrayRef<mlir::BlockArgument> args);

/// Verifies an EnclaveOp op.
///
/// This verifier checks the implicit requirement that the results of the
/// enclave are defined by the region terminator op.
///
/// @pre        `isa<EnclaveOp>(op)`
mlir::LogicalResult verify(mlir::Operation *op);

} // namespace enclave_interface_defaults

} // namespace amd::xten_nn

//===- Generated includes -------------------------------------------------===//

#include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterface.h.inc"

//===----------------------------------------------------------------------===//
