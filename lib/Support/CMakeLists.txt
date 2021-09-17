

if(${TORCH_FOUND})
  include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
  )

  set(ATEN_LIBS ${TORCH_LIBRARIES})
  add_compile_definitions(AIR_ENABLE_TORCH)
endif()

add_definitions("-Wno-string-conversion")
add_definitions("-Wno-covered-switch-default")

add_library(aten_ops SHARED
  aten_ops.cpp
)

install(TARGETS aten_ops
LIBRARY DESTINATION lib COMPONENT python
ARCHIVE DESTINATION lib COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)

target_link_libraries(aten_ops
  ${ATEN_LIBS}
  pthread
 )
