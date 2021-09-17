configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  _xten
  )

add_lit_testsuite(check-xten-python "Running the xten-python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-xten-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR_PYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-all check-xten-python)
