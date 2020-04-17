# ------------------------------------------------------------------------------
# Set NVCC flags
# ------------------------------------------------------------------------------
set(DF_CUDA_ARCH Auto CACHE STRING "A list of CUDA architectures to compile for. Specifying 'Auto' will attempt to autodetect available GPU devices")
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS ${DF_CUDA_ARCH})
set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math)
message("Compiling for CUDA architectures: ${CUDA_NVCC_ARCH_FLAGS}")

# suppress NVCC warning triggered in Sophus:
# "__device__ annotation is ignored on a function that is explicitly defaulted on its first declaration"
# see: https://github.com/kokkos/kokkos/issues/1473
list(APPEND CMAKE_CUDA_FLAGS -Xcudafe;--diag_suppress=esa_on_defaulted_function_ignored)

# append debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
  #list(APPEND CMAKE_CUDA_FLAGS --device-debug;--debug;-Xcompiler;-rdynamic;)#--ptxas-options=-v)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  list(APPEND CMAKE_CUDA_FLAGS -g;-lineinfo)
endif()
string(REPLACE ";" " " _TMP_STR "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${_TMP_STR}")

# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(cuda_sources
  cu_sfmaligner.cpp
  cu_image_proc.cpp
  cuda_context.cpp
  cu_se3aligner.cpp
  cu_depthaligner.cpp
)
set(cuda_headers
  kernel_utils.h
  cuda_context.h
  synced_pyramid.h
  cu_sfmaligner.h
  launch_utils.h
  cu_se3aligner.h
  cu_depthaligner.h
  device_info.h
  cu_image_proc.h
  reduction_items.h
)

set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_cuda SHARED ${cuda_sources} ${cuda_headers})
target_include_directories(df_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(df_cuda PUBLIC
  df_common
  Sophus::Sophus
  VisionCore
  Eigen3::Eigen
  cuda
  glog
)

#target_compile_features(df_cuda PUBLIC cxx_std_11)
set_property(TARGET df_cuda PROPERTY CUDA_STANDARD 11)
set_target_properties(df_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(df_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(df_cuda PROPERTIES RELOCATABLE_DEVICE_CODE ON)
