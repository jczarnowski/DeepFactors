# ------------------------------------------------------------------------------
# Dependencies
# For the downloaded deps, we rely here on XXX_DIR cache variables
# that were set by thirdparty/CMakeLists.txt file parsed earlier
# ------------------------------------------------------------------------------
find_package(CameraDrivers REQUIRED)

# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(drivers_sources
  # camera
  camera/openni_interface.cpp
  # dataset
  dataset/file_interface.cpp
  dataset/icl_interface.cpp
  dataset/scannet_interface.cpp
  dataset/tum_interface.cpp
  # top-level
  camera_interface_factory.cpp
)
set(drivers_headers
  # camera
  camera/live_interface.h
  camera/openni_interface.h
  # dataset
  dataset/dataset_interface.h
  dataset/file_interface.h
  dataset/icl_interface.h
  dataset/scannet_interface.h
  dataset/tum_interface.h
  # top-level
  camera_interface.h
  camera_interface_factory.h
)

if (DF_WITH_FLYCAP)
    list(APPEND drivers_sources camera/pointgrey_interface.cpp)
    list(APPEND drivers_headers camera/pointgrey_interface.h)
endif()

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_drivers SHARED ${drivers_sources} ${drivers_headers})
target_include_directories(df_drivers PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/camera
  ${CMAKE_CURRENT_SOURCE_DIR}/dataset
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(df_drivers PUBLIC
  df_common
  CameraDrivers
  boost_filesystem
  glog
)
