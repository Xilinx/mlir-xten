// (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

//===- XTenMinimizeLiveTensors.cpp -----------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This pass reorders operations to minimize the total size of live feature map
// tensors.

#include "PassDetail.h"

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Util/Util.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>
#include <set>

#define DEBUG_TYPE "xten-minimize-live"

using namespace mlir;
using namespace xilinx::xten;

namespace {

/// FM tensor sizes for a single operation.
struct OpSizes {
  /// Total size of the FM operands.
  size_t operands;
  /// Total size of the FM results.
  size_t results;
  /// Total size needed when executing, allowing for memory sharing.
  size_t running;
};

/// Information about memory requirements if the ops on a dependence
/// branch are executed.
/// Note that ops with multiple predecessors decide the order
/// of predecessor execution before they are added to the running
/// information, so once available it should be correct.
struct BranchRunning {
  /// The max size needed to run any operation on this branch.
  size_t maxRunning = 0;
  /// The size of the results of the last op in this branch.
  size_t lastResults = 0;
  /// The last op in this branch.
  Operation *lastOp{};
};

/// Various analysis results about an operation.
struct OpInfo {
  /// The operand this node represents.
  Operation *const op;
  /// The feature map operand values of this node.
  SmallVector<Value> const operands;
  /// The feature map results this node.
  SmallVector<Value> const results;
  /// The value that will share memory with the result during execution, if any.
  std::optional<Value> const sharesResultMemory = {};
  /// The consumers of any results. Note: this is filled progressively while
  /// collecting the operations.
  SmallVector<Operation *> consumers;
  /// Cumulative sizes of the FM tensors.
  OpSizes sizes = {};
  /// The preferred order of producers of the operands. Note: this is filled
  /// by post-analysis of the branching information.
  SmallVector<Operation *> orderedProducers;
  /// True when the operation has been moved in the IR.
  bool ordered = false;
};

mlir::LogicalResult verifyStrAttr(mlir::Operation *op, llvm::StringRef attrKey,
                                  llvm::StringRef attrValue) {
  if (!op->hasAttr(attrKey) ||
      !(op->getAttr(attrKey).cast<StringAttr>().str() == attrValue)) {
    return failure();
  }
  return success();
}

bool isInCoreChain(Operation *op) {
  return verifyStrAttr(op, "Reason", "InCoreChain").succeeded();
}

Operation *getInCoreChainOutputOp(Operation *op) {
  assert(isInCoreChain(op) && "Unexpected op");
  auto subgraphOp = cast<amd::xten_nn::SubgraphOp>(op);
  amd::xten_nn::OutputOp outputOp =
      *subgraphOp.getOps<amd::xten_nn::OutputOp>().begin();
  Value outputOperand = outputOp->getOperand(0);
  return outputOperand.getDefiningOp();
}

bool isAnySourceOp(mlir::Operation *op) {
  return verifyStrAttr(op, "Reason", "SourceOp").succeeded();
}

bool isSourceOp(mlir::Operation *op, StringRef opName) {
  return isAnySourceOp(op) && verifyStrAttr(op, "SourceOp", opName).succeeded();
}

bool isAnyPseudoOp(mlir::Operation *op) {
  return verifyStrAttr(op, "Reason", "PseudoOp").succeeded();
}

bool isPseudoOp(mlir::Operation *op, StringRef opName) {
  return isAnyPseudoOp(op) && verifyStrAttr(op, "Op", opName).succeeded();
}

bool isInterfaceOp(mlir::Operation *op) {
  return verifyStrAttr(op, "Reason", "Interface").succeeded();
}

bool isConcatSubgraph(Operation *op) {
  return isSourceOp(op, "onnx.Concat") || isPseudoOp(op, "Concat");
}

bool isTemplatedGraph(Operation *op) {
  return verifyStrAttr(op, "Reason", "TemplatedGraph").succeeded();
}

bool isGlobalAvgPool(Operation *op) {
  constexpr StringRef attr = "mllib_ops";
  if (!op->hasAttr(attr))
    return false;

  auto kernelName = cast<StringAttr>(op->getAttr(attr));
  return kernelName.strref().equals("GlobalAvgPool2D");
}

bool hasAllGAPInputs(Operation *op) {
  return llvm::all_of(op->getOperands(), [](Value operand) {
    Operation *producer = operand.getDefiningOp();
    if (producer == nullptr)
      return false;

    if (isInCoreChain(producer)) {
      producer = getInCoreChainOutputOp(producer);
    }

    return producer != nullptr && isGlobalAvgPool(producer);
  });
}

SmallVector<Value> getSubgraphIFMs(Operation *op) {

  // Handle IfmOperands attribute
  if (auto ifmIndices = op->getAttrOfType<ArrayAttr>("IfmOperands")) {

    // Get the operands from the values stored in IfmOperands Attr
    SmallVector<Value> ifmOperands;
    llvm::transform(ifmIndices.getAsValueRange<mlir::IntegerAttr>(),
                    std::back_inserter(ifmOperands), [&op](const APInt &idx) {
                      return op->getOperand(idx.getSExtValue());
                    });
    return ifmOperands;
  }
  return {};
}

std::optional<Value> getSubgraphOFM(Operation *op) {

  // Handle OfmShare attribute
  if (auto ofmShare = op->getAttrOfType<mlir::IntegerAttr>("OfmShare")) {
    return {op->getOperand(ofmShare.getInt())};
  }
  return {};
}

/// Returns the operand that will share memory with the result.
std::optional<Value> sharesMemoryWithResult(Operation *op) {

  if (isInCoreChain(op))
    return getSubgraphOFM(op);
  return {};
}

/// Returns all FM operands or fails if no IFM is known for the operation.
FailureOr<SmallVector<Value>> getFmOperands(Operation *op) {
  // No input to the function.
  if (isa<func::FuncOp>(op))
    return {{}};

  if (isInCoreChain(op))
    return {getSubgraphIFMs(op)};

  if (isConcatSubgraph(op))
    return {op->getOperands()};
  
  if (isTemplatedGraph(op))
    return {op->getOperands()};

  // Otherwise, this is a PseudoOp and IFM is the first operand.
  if (!(isAnyPseudoOp(op) || isInterfaceOp(op))) {
    op->emitError("Unknown operation");
    return failure();
  }
  return {{op->getOperand(0)}};
}

/// Return the size of the tensor type of \p val.
/// It is an error to call this with a non-tensor typed value.
size_t getSize(Value val) {
  auto type = dyn_cast<ShapedType>(val.getType());
  assert(type);

  // Get size in bytes
  assert(type.hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = type.getElementType();
  if (elementType.isIntOrFloat())
    return (elementType.getIntOrFloatBitWidth() * type.getNumElements()) / 8;

  if (auto complexType = elementType.dyn_cast<ComplexType>()) {
    elementType = complexType.getElementType();
    return (elementType.getIntOrFloatBitWidth() * type.getNumElements() * 2) / 8;
  }
  llvm_unreachable("Does not know how to compute size");
}

/// Debugging support - returns a simple name for an op.
StringRef getName(Operation *op) {
  if (op->hasAttr("layer_name"))
    return op->getAttrOfType<StringAttr>("layer_name").getValue();
  if (op->hasAttr("LayerName"))
    return op->getAttrOfType<StringAttr>("LayerName").getValue();
  return op->getName().getStringRef();
}

/// Debugging support - returns a string with all the op names.
std::string toStr(SmallVector<Operation *> const &vec) {
  std::string str("(");
  for (auto *op : vec)
    str += std::string(::getName(op)) + " ";
  return str + ")";
}

/// Determine the in/out/running L2 memory needed per Fwd.
void setOpSizes(OpInfo &opInfo) {
  size_t outgoing = 0;
  for_each(opInfo.results,
           [&outgoing](Value val) { outgoing += getSize(val); });
  opInfo.sizes.results = outgoing;
  size_t incoming = 0;
  for_each(opInfo.operands,
           [&incoming](Value val) { incoming += getSize(val); });
  opInfo.sizes.operands = incoming;
  opInfo.sizes.running = incoming;
  if (!opInfo.sharesResultMemory)
    opInfo.sizes.running += outgoing;
}

class XTenMinimizeLiveTensorsPass
    : public XTenMinimizeLiveTensorsBase<XTenMinimizeLiveTensorsPass> {
public:
  XTenMinimizeLiveTensorsPass() = default;
  XTenMinimizeLiveTensorsPass(const XTenMinimizeLiveTensorsPass &pass) =
      default;

  // Recursively collect the OpInfo of all FM producers.
  [[nodiscard]] LogicalResult
  collectOperandInfo(OpInfo const &opInfo) { // NOLINT(misc-no-recursion)
    // Visit all FM operands to collect their OpInfo.
    for (auto operand : opInfo.operands) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp == nullptr) {
        // Use currFn as stand-in for BlockArguments.
        assert(operand.isa<BlockArgument>());
        defOp = currFn;
      }

      // If OpInfo is already created, so we only need to note this consumer.
      auto prevInfoIt = opToInfo.find(defOp);
      if (prevInfoIt != opToInfo.end()) {
        prevInfoIt->second.consumers.push_back(opInfo.op);
        continue;
      }

      // Create the new OpInfo for the operand.
      FailureOr<SmallVector<Value>> const fmOperands = getFmOperands(defOp);
      if (failed(fmOperands)) {
        return LogicalResult::failure();
      }
      SmallVector<Value> fmResults;
      if (defOp != currFn) {
        fmResults = defOp->getResults();
      } else {
        fmResults = SmallVector<Value>(currFn.getBody().front().getArguments());
      }
      std::optional<Value> const sharesResultMemory = sharesMemoryWithResult(defOp);
      OpInfo info = {.op = defOp,
                     .operands = *fmOperands,
                     .results = fmResults,
                     .sharesResultMemory = sharesResultMemory,
                     .consumers = {opInfo.op}};
      auto [opFwdIt, succeeded] = opToInfo.emplace(defOp, std::move(info));
      setOpSizes(opFwdIt->second);

      // Recursively collect details of the operands of this operand.
      LogicalResult result = collectOperandInfo(opFwdIt->second);
      if (result.failed()) {
        signalPassFailure();
        return LogicalResult::failure();
      }
    }
    return LogicalResult::success();
  }

  /// Checks for illegal dead code by traversing operations and
  /// verifying if they are dead and any of their operands match any OpInfo.
  /// Dead code can be legal if it is not connected to operation collection
  /// through 'OpInfo' data structure.
  bool hasIllegalDeadCode() {
    return llvm::any_of(currFn.getBody().getOps(), [&](Operation &op) {
      if (opToInfo.find(&op) != opToInfo.end())
        return false; // okay as we schedule this op

      return llvm::any_of(op.getOperands(), [&](Value value) {
        // okay so long as it doesn't use the output of a scheduled op
        auto *defOp = value.getDefiningOp();
        if (defOp != nullptr && opToInfo.find(defOp) != opToInfo.end()) {
          op.emitError("Illegal operation");
          return true;
        }
        return false;
      });
    });
  }

  /// Recursively determine branch running sizes.
  ///
  /// \p opInfo points to the info for the op being analyzed.
  /// \p brInfo incoming branch info, to be updated by this op.
  /// \p completed contains the BranchRunning info for any fully
  ///    analyzed operations.
  void determineBranchRunning(OpInfo *opInfo, // NOLINT(misc-no-recursion)
                              BranchRunning &brInfo,
                              std::map<Operation *, BranchRunning> &completed) {
    // Analyze simple fallthrough operations.
    while (opInfo->operands.size() < 2 && opInfo->consumers.size() < 2) {
      // Avoid recomputing BranchRunning for operations we already visited.
      auto opIt = completed.find(opInfo->op);
      if (opIt != completed.end())
        return;

      brInfo.maxRunning = std::max(brInfo.maxRunning, opInfo->sizes.results);
      brInfo.lastResults = opInfo->sizes.results;
      opInfo->orderedProducers = {brInfo.lastOp};
      brInfo.lastOp = opInfo->op;
      completed.insert({opInfo->op, brInfo});

      if (opInfo->consumers.empty())
        return; // Nothing more to compute.
      opInfo = &opToInfo.at(opInfo->consumers.front());
    }

    // Avoid recomputing BranchRunning for operations we already visited.
    auto opIt = completed.find(opInfo->op);
    if (opIt != completed.end())
      return;

    // At a joining point - collect this branch and proceed iff all incoming
    // branches have been collected.
    SmallVector<BranchRunning> branches;
    for (Value val : opInfo->operands) {
      Operation *op = val.getDefiningOp();
      if (op == nullptr)
        op = currFn; // BlockArgument stand-in.

      auto opIt = completed.find(op);
      if (opIt == completed.end())
        return; // Another producer needs to complete first.
      branches.push_back(opIt->second);
    }

    // Concat producers should be ordered according to their operands
    // (op0 before op1 before op2, etc.), so no need to sort them out.
    if (!isConcatSubgraph(opInfo->op)) {
      std::sort(branches.begin(), branches.end(),
                [](BranchRunning &aBranch, BranchRunning &bBranch) -> bool {
                  return (aBranch.maxRunning - aBranch.lastResults) >
                         (bBranch.maxRunning - bBranch.lastResults);
                });
    } else if (hasAllGAPInputs(opInfo->op)) {
      // For GAP concats we want right-to-left ordering of the inputs
      std::reverse(branches.begin(), branches.end());
    }

    opInfo->orderedProducers.clear();
    for (BranchRunning const &branch : branches)
      opInfo->orderedProducers.push_back(branch.lastOp);

    // Complete the brInfo for this operation.
    size_t maxRunning = opInfo->sizes.running;
    for (BranchRunning const &branch : branches)
      maxRunning = std::max(maxRunning, branch.maxRunning);
    brInfo.maxRunning = maxRunning;
    brInfo.lastResults = opInfo->sizes.results;
    brInfo.lastOp = opInfo->op;
    completed.insert({opInfo->op, brInfo});

    // Continue to all consumers.
    for (Operation *consumer : opInfo->consumers) {
      BranchRunning nextRunning{.maxRunning = maxRunning,
                                .lastResults = opInfo->sizes.results,
                                .lastOp = opInfo->op};
      determineBranchRunning(&opToInfo.at(consumer), nextRunning, completed);
    }
    assert(opInfo->orderedProducers.size() == opInfo->operands.size());
  }

  /// Move the operators to the desired lexical order.
  void moveToOrder(OpInfo const &fwd) { // NOLINT(misc-no-recursion)
    for (Operation *op : fwd.orderedProducers) {
      if (op == currFn)
        continue; // BlockArguments cannot be moved.
      OpInfo &visitFwd = opToInfo.at(op);
      if (visitFwd.ordered)
        continue;

      op->moveBefore(fwd.op);
      visitFwd.ordered = true;
      moveToOrder(visitFwd);
    }
  }

  void runOnOperation() override {
    auto fwdFn = getOperation();
    mlir::Region &body = fwdFn.getBody();
    if (!body.hasOneBlock()) {
      fwdFn.emitError("function has complex control flow, aborting");
      signalPassFailure();
      return;
    }
    currFn = fwdFn;

    // A single block is expected. Building the initial graph starts from the
    // return and is successful when all FMs are ultimately produced from the
    // function arguments.
    Operation *returnStmt = body.begin()->getTerminator();
    assert(isa<func::ReturnOp>(returnStmt) &&
           "A function must terminate with a return stmt");
    SmallVector<Value> const retVal = returnStmt->getOperands();
    OpInfo fwdInfo = {.op = returnStmt, .operands = retVal, .results = {}};
    auto [opFwdIt, succeeded] =
        opToInfo.emplace(returnStmt, std::move(fwdInfo));
    OpInfo const &retFwd = opFwdIt->second;

    if (collectOperandInfo(retFwd).failed()) {
      signalPassFailure();
      return;
    }

    auto prevFwdIt = opToInfo.find(currFn);
    if (prevFwdIt == opToInfo.end()) {
      returnStmt->emitError(
          "function entry is not reached from the return stmt");
      signalPassFailure();
      return;
    }
    OpInfo &fnFwd = prevFwdIt->second;

    // Checks for any illegal dead code in currFn
    if (hasIllegalDeadCode()) {
      fwdFn->emitError("function cannot be rescheduled due to illegal dead "
                       "code, aborting.\n");
      signalPassFailure();
      return;
    }

    BranchRunning nextRunning{.maxRunning = fnFwd.sizes.running,
                              .lastResults = fnFwd.sizes.results,
                              .lastOp = fnFwd.op};
    std::map<Operation *, BranchRunning> completed;
    determineBranchRunning(&fnFwd, nextRunning, completed);

    LLVM_DEBUG(print());

    if (this->printBeforeSched)
      dumpAsDOTInOrder();

    if (this->printSchedCost)
      dumpAsDOT();

    // Reorder operations
    moveToOrder(retFwd);

    if (this->printAfterSched)
      dumpAsDOTInOrder();
  }

private:
  /// The analysis results for each operation.
  std::map<Operation *, OpInfo> opToInfo;
  /// The function being analyzed - needed in places to represent
  /// BlockArguments.
  mlir::func::FuncOp currFn;

  /// Debugging support - print some details of all .
  void print() {
    for (auto &[op, info] : opToInfo) {
      llvm::errs() << ::getName(op)
                   << " producers: " << toStr(info.orderedProducers)
                   << " consumers: " << toStr(info.consumers) << "\n";
    }
    llvm::errs() << "----\n";
  }

  /// Get a list of OpInfo's according to their order in the function
  std::vector<OpInfo> getInOrderOperationInfo() {

    std::vector<OpInfo> inOrderOpInfo;
    llvm::for_each(this->currFn.getRegion().getOps(), [&](Operation &op) {
      auto it = opToInfo.find(&op);
      if (it != opToInfo.end())
        inOrderOpInfo.push_back(it->second);
    });

    // Add function opInfo at last
    auto it = opToInfo.find(this->currFn);
    if (it != opToInfo.end())
      inOrderOpInfo.push_back(it->second);
    return inOrderOpInfo;
  }

  void dumpAsDOT() {
    dumpAsDOT(llvm::errs());
  }

  void dumpAsDOTInOrder() {
    dumpAsDOT(llvm::errs(), true);
  }

  void dumpAsDOT(raw_ostream &stream, bool printOrder = false) {

    mlir::raw_indented_ostream os(stream);

    os << "digraph G {\n";
    os.indent();
    os << "rankdir = TB     // top to bottom\n";
    os << "splines = spline // draw edges and route around nodes\n";
    os << "nodesep = 0.2    // horizontal compression\n";
    os << "ranksep = 0.5    // vertical compression\n";
    os << "node [shape=box] // default node style\n";
    os << "compound = true  // allow edges between subgraphs\n";

    auto startHTMLLabel = [&os]() {
      os << "<<table border=\"0\">\n";
      os.indent();
    };
    auto emitTableHeader = [&os](const std::string &str) {
      os << "<tr><td colspan=\"2\"><b>" << str << "</b></td></tr>\n";
    };
    auto emitTableRow = [&os](std::pair<std::string, std::string> &kv) {
      os << "<tr><td align=\"left\">" << std::get<0>(kv)
         << ":</td><td align=\"right\">" << std::get<1>(kv) << "</td></tr>\n";
    };
    auto endHTMLLabel = [&os]() {
      os.unindent();
      os << "</table>>";
    };

    os << "\n// Operations\n";
    os << "subgraph cost {\n";
    os.indent();

    // Get all OpInfo's in the order of which they appear in the function
    auto inOrderOperationInfos = getInOrderOperationInfo();

    // Record the name of which node for later reference
    mlir::DenseMap<Operation *, std::string> nodes;

    for (const auto &[idx, info] : llvm::enumerate(inOrderOperationInfos)) {
      Operation *op = info.op;

      std::string idxStr = std::to_string(idx);
      auto node = "op" + idxStr;
      nodes[op] = node;

      os << node << " [label = ";
      startHTMLLabel();
      emitTableHeader(("#" + idxStr + " " + ::getName(op)).str());
      std::pair<std::string, std::string> property(
          "Running Mem", std::to_string(info.sizes.running));
      emitTableRow(property);
      endHTMLLabel();
      os << "]\n";
    }

    if (!printOrder) {
      os << "\n// Dependences\n";
      // Ignore the last opInfo since it represents
      //  Arguments of the function
      for (auto &info :
           llvm::make_range(inOrderOperationInfos.begin(),
                            std::prev(inOrderOperationInfos.end()))) {
        Operation *op = info.op;
        for (const auto &consumer : info.consumers) {
          os << nodes[op] << " -> " << nodes[consumer] << " [";
          os << "label = ";
          startHTMLLabel();
          std::pair<std::string, std::string> property(
              "Mem", std::to_string(info.sizes.results));
          emitTableRow(property);
          endHTMLLabel();
          os << "]\n";
        }
      }

      // Draw edges that connect Function Arguments to Nodes
      auto &funcOpInfo = inOrderOperationInfos.back();
      llvm::SmallDenseSet<Operation *> visitedConsumers;
      for (const auto &consumer : funcOpInfo.consumers) {

        // One node can consume more multiple arguments
        // So only visit consumer once
        if (visitedConsumers.contains(consumer))
          continue;
        visitedConsumers.insert(consumer);

        // Get each argument value that contribute to
        // the running mem of the consumer
        llvm::for_each(consumer->getOperands(), [&](const Value &val) {
          if (val.getDefiningOp() == nullptr) {
            os << nodes[funcOpInfo.op] << " -> " << nodes[consumer] << " [";
            os << "label = ";
            startHTMLLabel();
            std::pair<std::string, std::string> property(
                "Mem", std::to_string(getSize(val)));
            emitTableRow(property);
            endHTMLLabel();
            os << "]\n";
          }
        });
      }
    }

    if (printOrder) {
      os << "\n// Order\n";
      for (auto [idx, info] : llvm::enumerate(
               llvm::make_range(inOrderOperationInfos.begin(),
                                std::prev(inOrderOperationInfos.end())))) {
        Operation *op = info.op;
        Operation *nextOp = inOrderOperationInfos[idx + 1].op;
        os << nodes[op] << " -> " << nodes[nextOp] << "\n";
      }
    }
    os.unindent();
    os << "}\n";

    os.unindent();
    os << "}\n";
  }
};

} // namespace

namespace xilinx::xten {

std::unique_ptr<OperationPass<func::FuncOp>>
createXTenMinimizeLiveTensorsPass() {
  return std::make_unique<XTenMinimizeLiveTensorsPass>();
}

} // namespace xilinx::xten
