import sys
import json

def _load_extension():
  import ctypes
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
  import _xten
  sys.setdlopenflags(flags)

  from mlir._cext_loader import _cext
  _cext.globals.append_dialect_search_prefix("acap.dialects")
  return _xten

_cext = _load_extension()
_cext._register_all_passes()
