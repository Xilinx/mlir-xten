

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
)
add_library(_acap SHARED
  AcapModule.cpp
)
set_target_properties(_acap PROPERTIES PREFIX "")

target_link_libraries(_acap
  PRIVATE
  ATenTransformPasses
  ATenConversionPasses
  ATenUtil
  XTenDialect
  ${Python3_LIBRARIES}

)

install(TARGETS _acap
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

install(DIRECTORY acap
DESTINATION python COMPONENT python)

#add_subdirectory(test)