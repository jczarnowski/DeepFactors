add_subdirectory(googletest)

# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(test_sources
  main.cpp
  ut_decoder.cpp
  ut_cuda_utils.cpp
  ut_se3aligner.cpp
  ut_sfmaligner.cpp
  ut_warping.cpp
  ut_pinhole_camera.cpp
  ut_nearestpsd.cpp
  ut_sampling.cpp
)

set(test_headers random_machine.h testing_utils.h)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_executable(ut_df ${test_sources} ${test_headers})
target_include_directories(ut_df PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ut_df PUBLIC
  gtest
  glog
  df_cuda
  df_core
  Eigen3::Eigen
  VisionCore
)
