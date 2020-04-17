# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(common_sources
  image_sequences.cpp
  timing.cpp
  logutils.cpp
  display_utils.cpp
  interp.cpp
)
set(common_headers
  # algorithm
  algorithm/pinhole_camera_impl.h
  algorithm/m_estimators.h
  algorithm/dense_sfm.h
  algorithm/camera_pyramid.h
  algorithm/warping.h
  algorithm/pinhole_camera.h
  algorithm/nearest_psd.h
  algorithm/lucas_kanade_se3.h
  # top-level
  image_sequences.h
  indexed_map.h
  tum_io.h
  interp.h
  timing.h
  display_utils.h
  logutils.h
  intrinsics.h
)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_common SHARED ${common_sources} ${common_headers})
target_include_directories(df_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
)
target_link_libraries(df_common PUBLIC
   Eigen3::Eigen
   Sophus::Sophus
   VisionCore
   opencv_core
   opencv_highgui
   jsoncpp
)
