

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
)
add_library(_xten SHARED
  XTenModule.cpp
)
set_target_properties(_xten PROPERTIES PREFIX "")

target_link_libraries(_xten
  PRIVATE
  XTenTransformPasses
  XTenConversionPasses
  XTenUtil
  XTenDialect
  ${Python3_LIBRARIES}

)

install(TARGETS _xten
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

install(DIRECTORY xten
DESTINATION python COMPONENT python)

#add_subdirectory(test)