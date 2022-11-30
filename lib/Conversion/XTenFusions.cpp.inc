// This file is the copy of the generated XtenFusions.cpp.inc (generated from
// XtenFusions.td by tablegen) with an extra setLayerNameAttr function for
// setting layer_name attribute for all Xten operations after the fusion. It
// was not possible to do this using table-driven rewrite rules because
// layer_name attribute has not defined in any of the definition of the Xten
// operations. Therefore, it was not possible to capture this attribute in the
// table gen file.

void setLayerNameAttr(Operation *source, Operation *target) {
  const std::string attrName = "layer_name";
  if (source == nullptr || target == nullptr)
    return;
  if (source->hasAttr(attrName))
    target->setAttr(attrName, source->getAttr(attrName));
  else
    source->emitError("The operation is expected to have layer_name attribute");
}

static ::mlir::LogicalResult static_dag_matcher_0(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0,
    ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops,
    ::mlir::Operation::operand_range &g, ::mlir::Operation::operand_range &f,
    ::mlir::Operation::operand_range &e, ::mlir::Operation::operand_range &c,
    ::mlir::Operation::operand_range &b, ::mlir::Operation::operand_range &d,
    ::mlir::Operation::operand_range &a) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dOp>(op0);
  (void)castedOp1;
  if (!(castedOp1)) {
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not xilinx::xten::Conv2dOp type";
    });
  }
  a = castedOp1.getODSOperands(0);
  b = castedOp1.getODSOperands(1);
  c = castedOp1.getODSOperands(2);
  d = castedOp1.getODSOperands(3);
  e = castedOp1.getODSOperands(4);
  f = castedOp1.getODSOperands(5);
  g = castedOp1.getODSOperands(6);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0,
    ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops,
    ::mlir::Operation::operand_range &g, ::mlir::Operation::operand_range &f,
    ::mlir::Operation::operand_range &e, ::mlir::Operation::operand_range &c,
    ::mlir::Operation::operand_range &h, ::mlir::Operation::operand_range &b,
    ::mlir::Operation::operand_range &d, ::mlir::Operation::operand_range &a) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dTensorAddOp>(op0);
  (void)castedOp1;
  if (!(castedOp1)) {
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not xilinx::xten::Conv2dTensorAddOp type";
    });
  }
  a = castedOp1.getODSOperands(0);
  b = castedOp1.getODSOperands(1);
  c = castedOp1.getODSOperands(2);
  d = castedOp1.getODSOperands(3);
  e = castedOp1.getODSOperands(4);
  f = castedOp1.getODSOperands(5);
  g = castedOp1.getODSOperands(6);
  h = castedOp1.getODSOperands(7);
  return ::mlir::success();
}

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:41
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("xten.add", 6, context,
                               {"xten.conv2d_tensoradd"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    xilinx::xten::Conv2dOp c2d1;
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    xilinx::xten::Conv2dOp c2d0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::AddOp>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dOp type";
        });
      }
      c2d0 = castedOp1;
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      c = castedOp1.getODSOperands(2);
      d = castedOp1.getODSOperands(3);
      e = castedOp1.getODSOperands(4);
      f = castedOp1.getODSOperands(5);
      g = castedOp1.getODSOperands(6);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 1 of castedOp0";
            });
      }
      auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dOp type";
        });
      }
      c2d1 = castedOp1;
      tblgen_ops.push_back(op1);
    }
    if (!((fuseFirstC2dInTensorAdd((*c2d0.getODSResults(0).begin()),
                                   (*c2d1.getODSResults(0).begin()))))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'c2d0, c2d1' failed to satisfy constraint: 'fuse "
                "first parameter into tensor add, otherwise fuse second.'";
      });
    }

    // Rewrite
    auto odsLoc =
        rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(),
                              tblgen_ops[2]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddOp tblgen_Conv2dTensorAddOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*c2d1.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops[1], tblgen_Conv2dTensorAddOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:46
*/
struct GeneratedConvert1 : public ::mlir::RewritePattern {
  GeneratedConvert1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("xten.add", 5, context,
                               {"xten.conv2d_tensoradd"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    xilinx::xten::Conv2dOp c2d0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::AddOp>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dOp type";
        });
      }
      c2d0 = castedOp1;
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 1 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, g, f,
                                              e, c, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc =
        rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(),
                              tblgen_ops[2]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddOp tblgen_Conv2dTensorAddOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*c2d0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops[2], tblgen_Conv2dTensorAddOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:51
*/
struct GeneratedConvert2 : public ::mlir::RewritePattern {
  GeneratedConvert2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("xten.add", 2, context,
                               {"xten.conv2d_tensoradd"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::AddOp>(op0);
    (void)castedOp0;
    h = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 1 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, g, f,
                                              e, c, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddOp tblgen_Conv2dTensorAddOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dTensorAddOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:54
*/
struct GeneratedConvert3 : public ::mlir::RewritePattern {
  GeneratedConvert3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("xten.add", 2, context,
                               {"xten.conv2d_tensoradd"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::AddOp>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, g, f,
                                              e, c, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    h = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddOp tblgen_Conv2dTensorAddOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dTensorAddOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:59
*/
struct GeneratedConvert4 : public ::mlir::RewritePattern {
  GeneratedConvert4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.relu", 2, context,
                               {"xten.conv2d_tensoradd_relu"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::torch::Torch::AtenReluOp>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, g, f,
                                              e, c, h, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddReLUOp tblgen_Conv2dTensorAddReLUOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddReLUOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddReLUOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddReLUOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dTensorAddReLUOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:62
*/
struct GeneratedConvert5 : public ::mlir::RewritePattern {
  GeneratedConvert5(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.leaky_relu", 2, context,
                               {"xten.conv2d_tensoradd_lrelu"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range alpha(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenLeakyReluOp>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, g, f,
                                              e, c, h, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    alpha = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddLReLUOp tblgen_Conv2dTensorAddLReLUOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*alpha.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddLReLUOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddLReLUOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddLReLUOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dTensorAddLReLUOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:66
*/
struct GeneratedConvert6 : public ::mlir::RewritePattern {
  GeneratedConvert6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("xten.globalaveragepool2d", 2, context,
                               {"xten.conv2d_tensoradd_globalaveragepool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::GlobalAveragePool2D>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      if (::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, g, f,
                                              e, c, h, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddGlobalAveragePoolOp
        tblgen_Conv2dTensorAddGlobalAveragePoolOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddGlobalAveragePoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddGlobalAveragePoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddGlobalAveragePoolOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(),
                     tblgen_Conv2dTensorAddGlobalAveragePoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:69
*/
struct GeneratedConvert7 : public ::mlir::RewritePattern {
  GeneratedConvert7(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern(
            "xten.globalaveragepool2d", 2, context,
            {"xten.conv2d_tensoradd_relu_globalaveragepool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::GlobalAveragePool2D>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      auto castedOp1 =
          ::llvm::dyn_cast<xilinx::xten::Conv2dTensorAddReLUOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dTensorAddReLUOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      c = castedOp1.getODSOperands(2);
      d = castedOp1.getODSOperands(3);
      e = castedOp1.getODSOperands(4);
      f = castedOp1.getODSOperands(5);
      g = castedOp1.getODSOperands(6);
      h = castedOp1.getODSOperands(7);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddReLUGlobalAveragePoolOp
        tblgen_Conv2dTensorAddReLUGlobalAveragePoolOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddReLUGlobalAveragePoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dTensorAddReLUGlobalAveragePoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddReLUGlobalAveragePoolOp_0.getODSResults(
                 0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(),
                     tblgen_Conv2dTensorAddReLUGlobalAveragePoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/XTenFusions.td:72
*/
struct GeneratedConvert8 : public ::mlir::RewritePattern {
  GeneratedConvert8(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern(
            "xten.globalaveragepool2d", 2, context,
            {"xten.conv2d_tensoradd_lrelu_globalaveragepool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range alpha(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<xilinx::xten::GlobalAveragePool2D>(op0);
    (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 0 of castedOp0";
            });
      }
      auto castedOp1 =
          ::llvm::dyn_cast<xilinx::xten::Conv2dTensorAddLReLUOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dTensorAddLReLUOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      c = castedOp1.getODSOperands(2);
      d = castedOp1.getODSOperands(3);
      e = castedOp1.getODSOperands(4);
      f = castedOp1.getODSOperands(5);
      g = castedOp1.getODSOperands(6);
      alpha = castedOp1.getODSOperands(7);
      h = castedOp1.getODSOperands(8);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dTensorAddLReLUGlobalAveragePoolOp
        tblgen_Conv2dTensorAddLReLUGlobalAveragePoolOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      tblgen_values.push_back((*d.begin()));
      tblgen_values.push_back((*e.begin()));
      tblgen_values.push_back((*f.begin()));
      tblgen_values.push_back((*g.begin()));
      tblgen_values.push_back((*alpha.begin()));
      tblgen_values.push_back((*h.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dTensorAddLReLUGlobalAveragePoolOp_0 =
          rewriter
              .create<xilinx::xten::Conv2dTensorAddLReLUGlobalAveragePoolOp>(
                  odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dTensorAddLReLUGlobalAveragePoolOp_0.getODSResults(
                 0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(),
                     tblgen_Conv2dTensorAddLReLUGlobalAveragePoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<GeneratedConvert0>(patterns.getContext());
  patterns.add<GeneratedConvert1>(patterns.getContext());
  patterns.add<GeneratedConvert2>(patterns.getContext());
  patterns.add<GeneratedConvert3>(patterns.getContext());
  patterns.add<GeneratedConvert4>(patterns.getContext());
  patterns.add<GeneratedConvert5>(patterns.getContext());
  patterns.add<GeneratedConvert6>(patterns.getContext());
  patterns.add<GeneratedConvert7>(patterns.getContext());
  patterns.add<GeneratedConvert8>(patterns.getContext());
}
