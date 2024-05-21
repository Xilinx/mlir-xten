# xten/__init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# (c) Copyright 2021 Xilinx, Inc. All Rights reserved.
# (c) Copyright 2022 - 2024 Advanced Micro Devices, Inc. All Rights reserved.

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
