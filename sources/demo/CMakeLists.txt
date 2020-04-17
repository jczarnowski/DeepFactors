# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(demo_sources
  main.cpp
  live_demo.cpp
)
set(demo_headers
  live_demo.h
)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_executable(df_demo ${demo_sources} ${demo_headers})
target_include_directories(df_demo PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(df_demo PUBLIC
  df_core
  df_gui
  df_drivers
  df_common
  opencv_core
  opencv_highgui
  boost_filesystem
  gflags
  glog
)
