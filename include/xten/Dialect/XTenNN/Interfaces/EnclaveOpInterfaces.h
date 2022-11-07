/// Declares the DLNN EnclaveOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "xten/Dialect/XTenNN/Enums.h"

namespace amd::xten_nn {

namespace enclave_interface_defaults {

/// Populates @p result with mappings from captured values to block arguments.
///
/// @pre        `isa<EnclaveOp>(op)`
void populateCaptureMap(Operation *op, BlockAndValueMapping &result);

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
void capture(Operation *op, ValueRange values, BlockAndValueMapping &result);

/// Removes the dead captures corresponding to @p args .
///
/// This default implementation assumes that operands are mapped 1-1 to block
/// arguments. It will therefore remove operands and arguments at the same
/// indices.
///
/// @pre        `isa<EnclaveOp>(op)`
/// @pre        @p args have no remaining uses
/// @pre        @p args are defined inside @p op
void uncapture(Operation *op, ArrayRef<BlockArgument> args);

/// Verifies an EnclaveOp op.
///
/// This verifier checks the implicit requirement that the results of the
/// enclave are defined by the region terminator op.
///
/// @pre        `isa<EnclaveOp>(op)`
LogicalResult verify(Operation *op);

} // namespace enclave_interface_defaults

} // namespace amd::xten_nn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/EnclaveOpInterface.h.inc"

//===----------------------------------------------------------------------===//