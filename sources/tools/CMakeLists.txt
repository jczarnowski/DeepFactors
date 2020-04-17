# ------------------------------------------------------------------------------
# decode_image
# Decode zero code for a given RGB image
# ------------------------------------------------------------------------------
add_executable(decode_image decode_image.cpp)
target_link_libraries(decode_image PUBLIC
  df_core
  df_common
  opencv_highgui
  opencv_imgproc
  opencv_core
  Eigen3::Eigen
  glog
  gflags
)

# ------------------------------------------------------------------------------
# kernel_benchmark
# Benchmarking threads/blocks for key CUDA kernels
# ------------------------------------------------------------------------------
add_executable(kernel_benchmark kernel_benchmark.cpp)
target_include_directories(kernel_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernel_benchmark PUBLIC
  df_cuda
  df_core
  df_common
  opencv_highgui
  opencv_core
  opencv_imgproc
  gflags
)

# ------------------------------------------------------------------------------
# result_viewer
# Display a ground-truth trajectory and reprojected depth from a dataset and
# an estimated trajectory 
# ------------------------------------------------------------------------------
add_executable(result_viewer result_viewer.cpp)
target_include_directories(result_viewer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(result_viewer PUBLIC
  df_gui
  df_drivers
  opencv_core
  opencv_highgui
  gflags
  glog
)

# ------------------------------------------------------------------------------
# voc_builder
# Builds a vocabulary of Brisk features extracted from a set of TUM sequences
# ------------------------------------------------------------------------------
add_executable(voc_builder voc_builder.cpp)
target_include_directories(voc_builder PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BRISK_INCLUDE_DIRS}
)
target_link_libraries(voc_builder PUBLIC
  df_drivers
  df_core # for FBrisk.cpp TODO
  opencv_core
  opencv_highgui
  opencv_features2d
  brisk
  gflags
  glog
  DBoW2
)

# ------------------------------------------------------------------------------
# voc_test
# Test a BoW vocabulary by calculating similarity between example images
# ------------------------------------------------------------------------------
add_executable(voc_test voc_test.cpp)
target_include_directories(voc_test PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BRISK_INCLUDE_DIRS}
)
target_link_libraries(voc_test PUBLIC
  df_core # for FBrisk.cpp TODO
  opencv_core
  opencv_highgui
  opencv_features2d
  brisk
  DBoW2
)


