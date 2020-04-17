cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# cmake_policy(SET CMP0060 NEW)

# ------------------------------------------------------------------------------
# Project name
# ------------------------------------------------------------------------------
project(deepfactors LANGUAGES CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.1)

# ------------------------------------------------------------------------------
# Default build configuration
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# output all executables to a single dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# ------------------------------------------------------------------------------
# CMake modules directory
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ------------------------------------------------------------------------------
# Build Options
# ------------------------------------------------------------------------------
option(DF_BUILD_TESTS "Build unit tests" ON)
option(DF_BUILD_DEMO "Build the main demo" ON)
option(DF_BUILD_TOOLS "Build tools" OFF)
option(DF_ENABLE_TIMING "Enable timing of crucial parts of the system" OFF)
option(DF_WITH_FLYCAP "Compile support for PoinGrey FlyCapture SDK cameras" OFF)

# Allow selecting the network code_size at compile-time
set(DF_CODE_SIZE 32 CACHE STRING "Size of the compact code representation")
set_property(CACHE DF_CODE_SIZE PROPERTY STRINGS 32 64 128)
add_definitions(-DDF_CODE_SIZE=${DF_CODE_SIZE})

if(DF_ENABLE_TIMING)
  add_definitions(-DENABLE_TIMING)
endif()

# ------------------------------------------------------------------------------
# Compile Options
# ------------------------------------------------------------------------------
# enable c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable warnings
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-catch-value>) # for pangolin

# disable eigen vectorization and alignment
# we were getting nasty segfaults heisenbugs with them
add_definitions(-DEIGEN_DONT_VECTORIZE 
                -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT 
                -DEIGEN_DONT_ALIGN)

# ------------------------------------------------------------------------------
# Print Project Info
# ------------------------------------------------------------------------------
message("Project: ${PROJECT_NAME}")
message("Version: ${${PROJECT_NAME}_VERSION}")
message("Build type: ${CMAKE_BUILD_TYPE}")
message("Compiled on: ${CMAKE_SYSTEM}")

# ------------------------------------------------------------------------------
# Dependencies
# For the downloaded deps, we rely here on XXX_DIR cache variables
# that were set by thirdparty/CMakeLists.txt file parsed earlier
# ------------------------------------------------------------------------------
add_subdirectory(thirdparty)
find_package(CUDA         QUIET REQUIRED)
find_package(Jsoncpp      QUIET REQUIRED)
find_package(Sophus       QUIET REQUIRED)
find_package(VisionCore   QUIET REQUIRED)
find_package(brisk        QUIET REQUIRED)
find_package(Glog         QUIET REQUIRED)
find_package(OpenCV       QUIET REQUIRED)
find_package(opengv       QUIET REQUIRED)
find_package(GTSAM        QUIET REQUIRED)
find_package(Eigen3       QUIET REQUIRED)
find_package(Pangolin     QUIET REQUIRED)
find_package(Boost        QUIET REQUIRED COMPONENTS filesystem)
find_package(TensorFlow   QUIET REQUIRED)

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_subdirectory(sources)

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------
if(DF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Installation - generate version file
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# ------------------------------------------------------------------------------
# Installation - export Targets
# ------------------------------------------------------------------------------
# export(EXPORT ${PROJECT_NAME}Targets
#   FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
#   NAMESPACE ${PROJECT_NAME}::)
# configure_file(${PROJECT_NAME}Config.cmake.in
#                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" COPYONLY)
#
# set(ConfigPackageLocation lib/cmake/${PROJECT_NAME})
# install(EXPORT ${PROJECT_NAME}Targets
#   FILE ${PROJECT_NAME}Targets.cmake
#   NAMESPACE ${PROJECT_NAME}::
#   DESTINATION ${ConfigPackageLocation})
# install(
#   FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#         "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#   DESTINATION ${ConfigPackageLocation}
# )

# export package to cmake cache
export(PACKAGE ${PROJECT_NAME})
