#include "mlir-c/Core.h"
#include "ATenDialect.h"

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace xilinx {
namespace aten {

void init(void) {
    mlir::registerDialect<xilinx::aten::ATenDialect>();
}

mlir_type_t make_list_type(mlir_type_t elemType)
{
  mlir::Type eTy = mlir::Type::getFromOpaquePointer(elemType);
  return mlir_type_t{ ATenListType::get(eTy).getAsOpaquePointer() };
}

} // namespace aten
} // namespace xilinx

PYBIND11_MODULE(pybind_aten, m) {
  m.doc() = "Python bindings for Xilinx ATen MLIR Dialect";
  m.def("version", []() { return "0.1"; });
  m.def("make_list_type", &xilinx::aten::make_list_type, "make ATenListType");
  m.def("init", &xilinx::aten::init, "register dialect");
}