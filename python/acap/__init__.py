import sys
import json

def _load_extension():
  import ctypes
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
  import _acap
  sys.setdlopenflags(flags)

  from mlir._cext_loader import _cext
  _cext.globals.append_dialect_search_prefix("acap.dialects")
  return _acap

_cext = _load_extension()
_cext._register_all_passes()

# affine_opt_tile_sizes = _cext.affine_opt_tile_sizes
# affine_opt_copy_depths = _cext.affine_opt_copy_depths
# affine_opt_copy_slow_space = _cext.affine_opt_copy_slow_space
# affine_opt_copy_fast_space = _cext.affine_opt_copy_fast_space

# # _simple_alloc_pass = _cext._simple_alloc_pass
# def simple_alloc_pass(mlir, model_json):
#   if model_json is None or not len(model_json):
#     model_json = "{}"
#   if not isinstance(model_json, str):
#     model_json = json.dumps(model_json)
#   return _simple_alloc_pass(mlir, model_json)