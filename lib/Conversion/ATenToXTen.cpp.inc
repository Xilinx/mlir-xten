// This file is the copy of the generated AtenToXten.cpp.inc (generated from
// AtenToXten.td by tablegen) with an extra setLayerNameAttr function for
// setting layer_name attribute for all Xten operations after the conversion. It
// was not possible to do this using table-driven rewrite rules because
// layer_name attribute has not defined in any of the definition of the Aten
// operations. Therefore, it was not possible to capture this attribute in the
// table gen file.

void setLayerNameAttr(Operation *source, Operation *target) {
  if (source == nullptr || target == nullptr)
    return;
  if (source->hasAttr("layer_name"))
    target->setAttr("layer_name", source->getAttr("layer_name"));
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
  auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dLReLUOp>(op0);
  (void)castedOp1;
  if (!(castedOp1)) {
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not xilinx::xten::Conv2dLReLUOp type";
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

static ::mlir::LogicalResult static_dag_matcher_2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0,
    ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops,
    ::mlir::Operation::operand_range &g, ::mlir::Operation::operand_range &f,
    ::mlir::Operation::operand_range &e, ::mlir::Operation::operand_range &c,
    ::mlir::Operation::operand_range &b, ::mlir::Operation::operand_range &d,
    ::mlir::Operation::operand_range &a) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dReLUOp>(op0);
  (void)castedOp1;
  if (!(castedOp1)) {
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not xilinx::xten::Conv2dReLUOp type";
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

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:36
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.convolution", 1, context,
                               {"xten.conv2d"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range groups(op0->getOperands());
    ::mlir::Operation::operand_range output_padding(op0->getOperands());
    ::mlir::Operation::operand_range transposed(op0->getOperands());
    ::mlir::Operation::operand_range dilation(op0->getOperands());
    ::mlir::Operation::operand_range stride(op0->getOperands());
    ::mlir::Operation::operand_range padding(op0->getOperands());
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::Operation::operand_range weight(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenConvolutionOp>(op0);
    (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    weight = castedOp0.getODSOperands(1);
    bias = castedOp0.getODSOperands(2);
    stride = castedOp0.getODSOperands(3);
    padding = castedOp0.getODSOperands(4);
    dilation = castedOp0.getODSOperands(5);
    transposed = castedOp0.getODSOperands(6);
    output_padding = castedOp0.getODSOperands(7);
    groups = castedOp0.getODSOperands(8);
    if (!((!llvm::dyn_cast<Torch::ConstantBoolOp>(
                (*transposed.begin()).getDefiningOp())
                .value()))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'transposed' failed to satisfy constraint: 'check to "
                "see if it is torch.constant.bool false'";
      });
    }
    if (!(((*output_padding.begin())
               .getDefiningOp<Torch::PrimListConstructOp>()
               .getNumOperands() == 0))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output_padding' failed to satisfy constraint: "
                "'check to see if it is torch.list is empty'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dOp tblgen_Conv2dOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*weight.begin()));
      tblgen_values.push_back((*bias.begin()));
      tblgen_values.push_back((*stride.begin()));
      tblgen_values.push_back((*padding.begin()));
      tblgen_values.push_back((*dilation.begin()));
      tblgen_values.push_back((*groups.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dOp_0 = rewriter.create<xilinx::xten::Conv2dOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_Conv2dOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:40
*/
struct GeneratedConvert1 : public ::mlir::RewritePattern {
  GeneratedConvert1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.relu", 2, context,
                               {"xten.conv2d_relu"}) {}
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
    xilinx::xten::Conv2dReLUOp tblgen_Conv2dReLUOp_0;
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
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dReLUOp_0 = rewriter.create<xilinx::xten::Conv2dReLUOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dReLUOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dReLUOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:43
*/
struct GeneratedConvert2 : public ::mlir::RewritePattern {
  GeneratedConvert2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.leaky_relu", 2, context,
                               {"xten.conv2d_lrelu"}) {}
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
    xilinx::xten::Conv2dLReLUOp tblgen_Conv2dLReLUOp_0;
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
      tblgen_Conv2dLReLUOp_0 = rewriter.create<xilinx::xten::Conv2dLReLUOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dLReLUOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dLReLUOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:46
*/
struct GeneratedConvert3 : public ::mlir::RewritePattern {
  GeneratedConvert3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.constant_pad_nd", 2, context,
                               {"xten.conv2d_lrelu_pad"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range pad(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenConstantPadNdOp>(op0);
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
    pad = castedOp0.getODSOperands(1);
    val = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dLReLUPadOp tblgen_Conv2dLReLUPadOp_0;
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
      tblgen_values.push_back((*pad.begin()));
      tblgen_values.push_back((*val.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dLReLUPadOp_0 =
          rewriter.create<xilinx::xten::Conv2dLReLUPadOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dLReLUPadOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dLReLUPadOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:49
*/
struct GeneratedConvert4 : public ::mlir::RewritePattern {
  GeneratedConvert4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.constant_pad_nd", 2, context,
                               {"xten.conv2d_relu_pad"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range pad(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenConstantPadNdOp>(op0);
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
      if (::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, g, f,
                                              e, c, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    pad = castedOp0.getODSOperands(1);
    val = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dReLUPadOp tblgen_Conv2dReLUPadOp_0;
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
      tblgen_values.push_back((*pad.begin()));
      tblgen_values.push_back((*val.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dReLUPadOp_0 = rewriter.create<xilinx::xten::Conv2dReLUPadOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dReLUPadOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dReLUPadOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:57
*/
struct GeneratedConvert5 : public ::mlir::RewritePattern {
  GeneratedConvert5(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.max_pool2d", 2, context,
                               {"xten.conv2d_lrelu_maxpool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range mp6(op0->getOperands());
    ::mlir::Operation::operand_range mp5(op0->getOperands());
    ::mlir::Operation::operand_range mp2(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range mp3(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range mp4(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenMaxPool2dOp>(op0);
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
    mp2 = castedOp0.getODSOperands(1);
    mp3 = castedOp0.getODSOperands(2);
    mp4 = castedOp0.getODSOperands(3);
    mp5 = castedOp0.getODSOperands(4);
    mp6 = castedOp0.getODSOperands(5);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dLReLUMaxPoolOp tblgen_Conv2dLReLUMaxPoolOp_0;
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
      tblgen_values.push_back((*mp2.begin()));
      tblgen_values.push_back((*mp3.begin()));
      tblgen_values.push_back((*mp4.begin()));
      tblgen_values.push_back((*mp5.begin()));
      tblgen_values.push_back((*mp6.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dLReLUMaxPoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dLReLUMaxPoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dLReLUMaxPoolOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dLReLUMaxPoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:61
*/
struct GeneratedConvert6 : public ::mlir::RewritePattern {
  GeneratedConvert6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.max_pool2d", 2, context,
                               {"xten.conv2d_lrelu_pad_maxpool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range mp5(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range h(op0->getOperands());
    ::mlir::Operation::operand_range mp6(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range mp4(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range mp3(op0->getOperands());
    ::mlir::Operation::operand_range pad(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range mp2(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenMaxPool2dOp>(op0);
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
      auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dLReLUPadOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dLReLUPadOp type";
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
      pad = castedOp1.getODSOperands(8);
      val = castedOp1.getODSOperands(9);
      tblgen_ops.push_back(op1);
    }
    mp2 = castedOp0.getODSOperands(1);
    mp3 = castedOp0.getODSOperands(2);
    mp4 = castedOp0.getODSOperands(3);
    mp5 = castedOp0.getODSOperands(4);
    mp6 = castedOp0.getODSOperands(5);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dLReLUPadMaxPoolOp tblgen_Conv2dLReLUPadMaxPoolOp_0;
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
      tblgen_values.push_back((*pad.begin()));
      tblgen_values.push_back((*val.begin()));
      tblgen_values.push_back((*mp2.begin()));
      tblgen_values.push_back((*mp3.begin()));
      tblgen_values.push_back((*mp4.begin()));
      tblgen_values.push_back((*mp5.begin()));
      tblgen_values.push_back((*mp6.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dLReLUPadMaxPoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dLReLUPadMaxPoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dLReLUPadMaxPoolOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dLReLUPadMaxPoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:70
*/
struct GeneratedConvert7 : public ::mlir::RewritePattern {
  GeneratedConvert7(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.max_pool2d", 2, context,
                               {"xten.conv2d_relu_maxpool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range mp6(op0->getOperands());
    ::mlir::Operation::operand_range mp5(op0->getOperands());
    ::mlir::Operation::operand_range mp2(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range mp3(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range mp4(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenMaxPool2dOp>(op0);
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
      if (::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, g, f,
                                              e, c, b, d, a))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    mp2 = castedOp0.getODSOperands(1);
    mp3 = castedOp0.getODSOperands(2);
    mp4 = castedOp0.getODSOperands(3);
    mp5 = castedOp0.getODSOperands(4);
    mp6 = castedOp0.getODSOperands(5);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dReLUMaxPoolOp tblgen_Conv2dReLUMaxPoolOp_0;
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
      tblgen_values.push_back((*mp2.begin()));
      tblgen_values.push_back((*mp3.begin()));
      tblgen_values.push_back((*mp4.begin()));
      tblgen_values.push_back((*mp5.begin()));
      tblgen_values.push_back((*mp6.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dReLUMaxPoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dReLUMaxPoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dReLUMaxPoolOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dReLUMaxPoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:74
*/
struct GeneratedConvert8 : public ::mlir::RewritePattern {
  GeneratedConvert8(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.max_pool2d", 2, context,
                               {"xten.conv2d_relu_pad_maxpool"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range mp6(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range mp4(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range mp3(op0->getOperands());
    ::mlir::Operation::operand_range pad(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range mp2(op0->getOperands());
    ::mlir::Operation::operand_range mp5(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenMaxPool2dOp>(op0);
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
      auto castedOp1 = ::llvm::dyn_cast<xilinx::xten::Conv2dReLUPadOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not xilinx::xten::Conv2dReLUPadOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      c = castedOp1.getODSOperands(2);
      d = castedOp1.getODSOperands(3);
      e = castedOp1.getODSOperands(4);
      f = castedOp1.getODSOperands(5);
      g = castedOp1.getODSOperands(6);
      pad = castedOp1.getODSOperands(7);
      val = castedOp1.getODSOperands(8);
      tblgen_ops.push_back(op1);
    }
    mp2 = castedOp0.getODSOperands(1);
    mp3 = castedOp0.getODSOperands(2);
    mp4 = castedOp0.getODSOperands(3);
    mp5 = castedOp0.getODSOperands(4);
    mp6 = castedOp0.getODSOperands(5);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dReLUPadMaxPoolOp tblgen_Conv2dReLUPadMaxPoolOp_0;
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
      tblgen_values.push_back((*pad.begin()));
      tblgen_values.push_back((*val.begin()));
      tblgen_values.push_back((*mp2.begin()));
      tblgen_values.push_back((*mp3.begin()));
      tblgen_values.push_back((*mp4.begin()));
      tblgen_values.push_back((*mp5.begin()));
      tblgen_values.push_back((*mp6.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dReLUPadMaxPoolOp_0 =
          rewriter.create<xilinx::xten::Conv2dReLUPadMaxPoolOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dReLUPadMaxPoolOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dReLUPadMaxPoolOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:78
*/
struct GeneratedConvert9 : public ::mlir::RewritePattern {
  GeneratedConvert9(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.relu", 3, context,
                               {"xten.conv2d_bn_relu"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range a7(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range d(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a2(op0->getOperands());
    ::mlir::Operation::operand_range e(op0->getOperands());
    ::mlir::Operation::operand_range f(op0->getOperands());
    ::mlir::Operation::operand_range g(op0->getOperands());
    ::mlir::Operation::operand_range a8(op0->getOperands());
    ::mlir::Operation::operand_range a6(op0->getOperands());
    ::mlir::Operation::operand_range a4(op0->getOperands());
    ::mlir::Operation::operand_range a1(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range a3(op0->getOperands());
    ::mlir::Operation::operand_range a5(op0->getOperands());
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
      auto castedOp1 =
          ::llvm::dyn_cast<::mlir::torch::Torch::AtenBatchNormOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::torch::Torch::AtenBatchNormOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)) {
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic
                                                                &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        if (::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, g, f,
                                                e, c, b, d, a))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      a1 = castedOp1.getODSOperands(1);
      a2 = castedOp1.getODSOperands(2);
      a3 = castedOp1.getODSOperands(3);
      a4 = castedOp1.getODSOperands(4);
      a5 = castedOp1.getODSOperands(5);
      a6 = castedOp1.getODSOperands(6);
      a7 = castedOp1.getODSOperands(7);
      a8 = castedOp1.getODSOperands(8);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc =
        rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(),
                              tblgen_ops[2]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::Conv2dBatchNormReLUOp tblgen_Conv2dBatchNormReLUOp_0;
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
      tblgen_values.push_back((*a1.begin()));
      tblgen_values.push_back((*a2.begin()));
      tblgen_values.push_back((*a3.begin()));
      tblgen_values.push_back((*a4.begin()));
      tblgen_values.push_back((*a5.begin()));
      tblgen_values.push_back((*a6.begin()));
      tblgen_values.push_back((*a7.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2dBatchNormReLUOp_0 =
          rewriter.create<xilinx::xten::Conv2dBatchNormReLUOp>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_Conv2dBatchNormReLUOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(tblgen_ops.back(), tblgen_Conv2dBatchNormReLUOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:86
*/
struct GeneratedConvert10 : public ::mlir::RewritePattern {
  GeneratedConvert10(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.add_.Tensor", 3, context,
                               {"xten.add_constant"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::FlatSymbolRefAttr ac;
    ::mlir::func::ConstantOp c;
    ::mlir::FlatSymbolRefAttr ab;
    ::mlir::func::ConstantOp b;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenAdd_TensorOp>(op0);
    (void)castedOp0;
    a = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 1 of castedOp0";
            });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::func::ConstantOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::func::ConstantOp type";
        });
      }
      b = castedOp1;
      {
        auto tblgen_attr =
            op1->getAttrOfType<::mlir::FlatSymbolRefAttr>("value");
        (void)tblgen_attr;
        if (!(tblgen_attr)) {
          return rewriter.notifyMatchFailure(
              op1, [&](::mlir::Diagnostic &diag) {
                diag << "expected op 'func.constant' to have attribute 'value' "
                        "of type '::mlir::FlatSymbolRefAttr'";
              });
        }
        ab = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 2 of castedOp0";
            });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::func::ConstantOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::func::ConstantOp type";
        });
      }
      c = castedOp1;
      {
        auto tblgen_attr =
            op1->getAttrOfType<::mlir::FlatSymbolRefAttr>("value");
        (void)tblgen_attr;
        if (!(tblgen_attr)) {
          return rewriter.notifyMatchFailure(
              op1, [&](::mlir::Diagnostic &diag) {
                diag << "expected op 'func.constant' to have attribute 'value' "
                        "of type '::mlir::FlatSymbolRefAttr'";
              });
        }
        ac = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc =
        rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(),
                              tblgen_ops[2]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::AddConstantOp tblgen_AddConstantOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddConstantOp_0 = rewriter.create<xilinx::xten::AddConstantOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_AddConstantOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_AddConstantOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:89
*/
struct GeneratedConvert11 : public ::mlir::RewritePattern {
  GeneratedConvert11(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.mm", 1, context, {"xten.mm"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::torch::Torch::AtenMmOp>(op0);
    (void)castedOp0;
    a = castedOp0.getODSOperands(0);
    b = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::MMOp tblgen_MMOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MMOp_0 = rewriter.create<xilinx::xten::MMOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_MMOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_MMOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:92
*/
struct GeneratedConvert12 : public ::mlir::RewritePattern {
  GeneratedConvert12(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.mul.Tensor", 1, context,
                               {"xten.mul"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenMulTensorOp>(op0);
    (void)castedOp0;
    a = castedOp0.getODSOperands(0);
    b = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::MulOp tblgen_MulOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_0 = rewriter.create<xilinx::xten::MulOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_MulOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_MulOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:96
*/
struct GeneratedConvert13 : public ::mlir::RewritePattern {
  GeneratedConvert13(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.add.Tensor", 1, context,
                               {"xten.add"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenAddTensorOp>(op0);
    (void)castedOp0;
    a = castedOp0.getODSOperands(0);
    b = castedOp0.getODSOperands(1);
    c = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::AddOp tblgen_AddOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddOp_0 = rewriter.create<xilinx::xten::AddOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_AddOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_AddOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:99
*/
struct GeneratedConvert14 : public ::mlir::RewritePattern {
  GeneratedConvert14(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten._softmax", 1, context,
                               {"xten.softmax"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::Aten_SoftmaxOp>(op0);
    (void)castedOp0;
    a = castedOp0.getODSOperands(0);
    b = castedOp0.getODSOperands(1);
    c = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::SoftmaxOp tblgen_SoftmaxOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*c.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SoftmaxOp_0 = rewriter.create<xilinx::xten::SoftmaxOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_SoftmaxOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_SoftmaxOp_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:107
*/
struct GeneratedConvert15 : public ::mlir::RewritePattern {
  GeneratedConvert15(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.mean.dim", 1, context,
                               {"xten.globalaveragepool2d"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range reduce(op0->getOperands());
    ::mlir::Operation::operand_range keepdim(op0->getOperands());
    ::mlir::Operation::operand_range dims(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::torch::Torch::AtenMeanDimOp>(op0);
    (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    dims = castedOp0.getODSOperands(1);
    keepdim = castedOp0.getODSOperands(2);
    reduce = castedOp0.getODSOperands(3);
    if (!((isReduceMeanGlobalAveragePool2D((*dims.begin()),
                                           (*keepdim.begin()))))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'dims, keepdim' failed to satisfy constraint: 'check "
                "if attributes are valid for global average pool'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::GlobalAveragePool2D tblgen_GlobalAveragePool2D_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GlobalAveragePool2D_0 =
          rewriter.create<xilinx::xten::GlobalAveragePool2D>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_GlobalAveragePool2D_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_GlobalAveragePool2D_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:115
*/
struct GeneratedConvert16 : public ::mlir::RewritePattern {
  GeneratedConvert16(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.avg_pool2d", 4, context,
                               {"xten.globalaveragepool2d"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::IntegerAttr cp;
    ::mlir::torch::Torch::ConstantBoolOp ceil_mode;
    ::mlir::IntegerAttr cm;
    ::mlir::Operation::operand_range padding(op0->getOperands());
    ::mlir::Operation::operand_range stride(op0->getOperands());
    ::mlir::Operation::operand_range kernel_size(op0->getOperands());
    ::mlir::torch::Torch::ConstantBoolOp count_padding;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenAvgPool2dOp>(op0);
    (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    kernel_size = castedOp0.getODSOperands(1);
    stride = castedOp0.getODSOperands(2);
    padding = castedOp0.getODSOperands(3);
    {
      auto *op1 = (*castedOp0.getODSOperands(4).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 4 of castedOp0";
            });
      }
      auto castedOp1 =
          ::llvm::dyn_cast<::mlir::torch::Torch::ConstantBoolOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::torch::Torch::ConstantBoolOp type";
        });
      }
      ceil_mode = castedOp1;
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::IntegerAttr>("value");
        (void)tblgen_attr;
        if (!(tblgen_attr)) {
          return rewriter.notifyMatchFailure(
              op1, [&](::mlir::Diagnostic &diag) {
                diag << "expected op 'torch.constant.bool' to have attribute "
                        "'value' of type '::mlir::IntegerAttr'";
              });
        }
        cm = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(5).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 5 of castedOp0";
            });
      }
      auto castedOp1 =
          ::llvm::dyn_cast<::mlir::torch::Torch::ConstantBoolOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::torch::Torch::ConstantBoolOp type";
        });
      }
      count_padding = castedOp1;
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::IntegerAttr>("value");
        (void)tblgen_attr;
        if (!(tblgen_attr)) {
          return rewriter.notifyMatchFailure(
              op1, [&](::mlir::Diagnostic &diag) {
                diag << "expected op 'torch.constant.bool' to have attribute "
                        "'value' of type '::mlir::IntegerAttr'";
              });
        }
        cp = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(6).begin()).getDefiningOp();
      if (!(op1)) {
        return rewriter.notifyMatchFailure(
            castedOp0, [&](::mlir::Diagnostic &diag) {
              diag
                  << "There's no operation that defines operand 6 of castedOp0";
            });
      }
      auto castedOp1 =
          ::llvm::dyn_cast<::mlir::torch::Torch::ConstantNoneOp>(op1);
      (void)castedOp1;
      if (!(castedOp1)) {
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::torch::Torch::ConstantNoneOp type";
        });
      }
      tblgen_ops.push_back(op1);
    }
    if (!((isAvgPoolGlobalAveragePool2D(
            (*input.begin()), (*kernel_size.begin()), (*stride.begin()),
            (*padding.begin()))))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input, kernel_size, stride, padding' failed to "
                "satisfy constraint: 'check if attributes of AvgPool2D are "
                "valid for global average pool'";
      });
    }
    if (!((!llvm::dyn_cast<Torch::ConstantBoolOp>(
                (*ceil_mode.getODSResults(0).begin()).getDefiningOp())
                .value()))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'ceil_mode' failed to satisfy constraint: 'check to "
                "see if it is torch.constant.bool false'";
      });
    }
    if (!((llvm::dyn_cast<Torch::ConstantBoolOp>(
               (*count_padding.getODSResults(0).begin()).getDefiningOp())
               .value()))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'count_padding' failed to satisfy constraint: 'check "
                "to see if it is torch.constant.bool true'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(),
         tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::GlobalAveragePool2D tblgen_GlobalAveragePool2D_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GlobalAveragePool2D_0 =
          rewriter.create<xilinx::xten::GlobalAveragePool2D>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_GlobalAveragePool2D_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_GlobalAveragePool2D_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:123
*/
struct GeneratedConvert17 : public ::mlir::RewritePattern {
  GeneratedConvert17(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.adaptive_avg_pool2d", 1, context,
                               {"xten.globalaveragepool2d"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range outsizes(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 =
        ::llvm::dyn_cast<::mlir::torch::Torch::AtenAdaptiveAvgPool2dOp>(op0);
    (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    outsizes = castedOp0.getODSOperands(1);
    if (!((isAdaptiveAvgPoolGlobalAveragePool2D((*outsizes.begin()))))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'outsizes' failed to satisfy constraint: 'check if "
                "attributes are valid for global average pool'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::GlobalAveragePool2D tblgen_GlobalAveragePool2D_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GlobalAveragePool2D_0 =
          rewriter.create<xilinx::xten::GlobalAveragePool2D>(
              odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_GlobalAveragePool2D_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_GlobalAveragePool2D_0);

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    /scratch/projects/vitis_flexml/third-party/mlir-xten/include/xten/Conversion/ATenToXTen.td:135
*/
struct GeneratedConvert18 : public ::mlir::RewritePattern {
  GeneratedConvert18(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("torch.aten.linear", 1, context,
                               {"xten.linear"}) {}
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::Operation::operand_range weights(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::torch::Torch::AtenLinearOp>(op0);
    (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    weights = castedOp0.getODSOperands(1);
    bias = castedOp0.getODSOperands(2);
    if (!((checkLinearForXten((*input.begin()), (*weights.begin()),
                              (*bias.begin()))))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input, weights, bias' failed to satisfy constraint: "
                "'Only specific ranks for tensors are supported.\nCheck the "
                "ranks of the inputs:\n    y = A*B^T + C\nWhere rank(A) == 2, "
                "rank(B) == 2 and rank(C) == 1.\nNote, C must be defined. It "
                "cannot be set to none'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()});
    (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    xilinx::xten::LinearOp tblgen_LinearOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values;
      (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*weights.begin()));
      tblgen_values.push_back((*bias.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types;
      (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_LinearOp_0 = rewriter.create<xilinx::xten::LinearOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
             tblgen_LinearOp_0.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }

    setLayerNameAttr(op0, tblgen_LinearOp_0);

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
  patterns.add<GeneratedConvert9>(patterns.getContext());
  patterns.add<GeneratedConvert10>(patterns.getContext());
  patterns.add<GeneratedConvert11>(patterns.getContext());
  patterns.add<GeneratedConvert12>(patterns.getContext());
  patterns.add<GeneratedConvert13>(patterns.getContext());
  patterns.add<GeneratedConvert14>(patterns.getContext());
  patterns.add<GeneratedConvert15>(patterns.getContext());
  patterns.add<GeneratedConvert16>(patterns.getContext());
  patterns.add<GeneratedConvert17>(patterns.getContext());
  patterns.add<GeneratedConvert18>(patterns.getContext());
}
