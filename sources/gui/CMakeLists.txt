# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(gui_sources
  visualizer.cpp
  keyframe_renderer.cpp
)
set(gui_headers
  visualizer.h
  keyframe_renderer.h
)

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_library(df_gui SHARED ${gui_sources} ${gui_headers})
target_include_directories(df_gui PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Pangolin_INCLUDE_DIR}
)
target_link_libraries(df_gui PUBLIC
  df_common
  df_core # for keyframe_map definition
  gflags
  glog
  ${Pangolin_LIBRARY}
)
target_compile_definitions(df_gui PUBLIC
  -DDF_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
