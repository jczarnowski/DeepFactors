# So this is a big hacky. Brisk leaks architecture specific
# calls in public headers, which forces us to compile for that specific architecture
# (for most cases, march=native will work). GT-SAM has problems with march=native
# due to eigen, so we isolate brisk in a separate library which we compile for
# the native architecture and build the rest of our system without it.
# For this to work, we need to make sure that the interface between the native and
# non-native optimised code is as simple as possible -- no eigen!
add_library(core_brisk_wrap SHARED system/fbrisk.cpp system/fbrisk.h)
target_compile_options(core_brisk_wrap PRIVATE -march=native -mtune=native)
target_link_libraries(core_brisk_wrap PUBLIC
    brisk
    DBoW2
    opencv_core)
target_include_directories(core_brisk_wrap PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/system
  ${BRISK_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(core_sources
  # gtsam
  gtsam/sparse_geometric_factor.cpp
  gtsam/photometric_factor.cpp
  gtsam/reprojection_factor.cpp
  gtsam/depth_prior_factor.cpp
  gtsam/uniform_sampler.cpp
  # system
  system/camera_tracker.cpp
  system/loop_detector.cpp
  # features
  features/matching.cpp
  # mapping
  mapping/work.cpp
  mapping/work_manager.cpp
  mapping/mapper.cpp
  mapping/df_work.cpp
  # network
  network/decoder_network.cpp
  # top-level
  deepfactors.cpp
  deepfactors_options.cpp
)

set(core_headers
  # gtsam
  gtsam/photometric_factor.h
  gtsam/reprojection_factor.h
  gtsam/gtsam_traits.h
  gtsam/sparse_geometric_factor.h
  gtsam/gtsam_utils.h
  gtsam/uniform_sampler.h
  gtsam/factor_graph.h
  gtsam/depth_prior_factor.h
  # system
  system/camera_tracker.h
  system/loop_detector.h
  # features
  features/feature_detection.h
  features/matching.h
  # mapping
  mapping/keyframe_map.h
  mapping/keyframe.h
  mapping/work_manager.h
  mapping/df_work.h
  mapping/mapper.h
  mapping/frame.h
  mapping/work.h
  # network
  network/tfwrap.h
  network/decoder_network.h
  # top-level
  deepfactors.h
  deepfactors_options.h
)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_core SHARED ${core_sources} ${core_headers})
target_include_directories(df_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/features
  ${CMAKE_CURRENT_SOURCE_DIR}/gtsam
  ${CMAKE_CURRENT_SOURCE_DIR}/mapping
  ${CMAKE_CURRENT_SOURCE_DIR}/network
  ${CMAKE_CURRENT_SOURCE_DIR}/system
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(df_core 
  PUBLIC
  df_common
  df_cuda
  Eigen3::Eigen
  VisionCore
  DBoW2
  gtsam
  opengv
  glog
  core_brisk_wrap
  PRIVATE
  TensorFlow
  jsoncpp
)

