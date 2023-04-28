# lit.cfg.py
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = 'XTen-Unit'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.cmake_binary_dir, 'test')
config.test_source_root = config.cmake_current_binary_dir

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, 'Tests')

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if 'TMP' in os.environ:
    config.environment['TMP'] = os.environ['TMP']
if 'TEMP' in os.environ:
    config.environment['TEMP'] = os.environ['TEMP']

# Propagate HOME as it can be used to override incorrect homedir in passwd
# that causes the tests to fail.
if 'HOME' in os.environ:
    config.environment['HOME'] = os.environ['HOME']

# Propagate sanitizer options.
for var in [
    'ASAN_SYMBOLIZER_PATH',
    'HWASAN_SYMBOLIZER_PATH',
    'MSAN_SYMBOLIZER_PATH',
    'TSAN_SYMBOLIZER_PATH',
    'UBSAN_SYMBOLIZER_PATH',
    'ASAN_OPTIONS',
    'HWASAN_OPTIONS',
    'MSAN_OPTIONS',
    'TSAN_OPTIONS',
    'UBSAN_OPTIONS',
]:
    if var in os.environ:
        config.environment[var] = os.environ[var]

# Win32 seeks DLLs along %PATH%.
if sys.platform in ['win32', 'cygwin'] and os.path.isdir(config.shlibdir):
    config.environment['PATH'] = os.path.pathsep.join((
            config.shlibdir, config.environment['PATH']))

# Win32 may use %SYSTEMDRIVE% during file system shell operations, so propogate.
if sys.platform == 'win32' and 'SYSTEMDRIVE' in os.environ:
    config.environment['SYSTEMDRIVE'] = os.environ['SYSTEMDRIVE']
