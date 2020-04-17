# ------------------------------------------------------------------------------
# Find all the cmake-based dependencies
# If you haven't built them, this should also be able to find other installations
# You can point find_package to them by setting e.g. the Sophus_DIR variable
# Note that we're using HINTS instead of PATHS to make sure to search the
# local build first before looking into cmake package registry
# See: https://cmake.org/cmake/help/v3.0/command/find_package.html
# ------------------------------------------------------------------------------
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_PREFIX_PATH ${INSTALL_DIR})
# Sophus needs to be before VisionCore as it looks for it
find_package(Sophus         NO_DEFAULT_PATH PATHS ${BUILD_DIR}/Sophus REQUIRED)
find_package(VisionCore     NO_DEFAULT_PATH PATHS ${BUILD_DIR}/vision_core REQUIRED)
find_package(brisk          PATHS ${INSTALL_DIR}/lib/CMake REQUIRED) # Stefan's "CMake" instead of "cmake"
find_package(CameraDrivers  QUIET REQUIRED)
find_package(DBoW2          QUIET REQUIRED)
find_package(GTSAM          QUIET REQUIRED)
find_package(opengv         QUIET REQUIRED)
find_package(Eigen3         QUIET REQUIRED)
find_package(Pangolin       QUIET REQUIRED)

## ------------------------------------------------------------------------------
## Modify brisk's imported target to add their include directories
## ------------------------------------------------------------------------------
#set_property(TARGET brisk APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
#  ${BRISK_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Add a DBoW2 imported target because it does something weird in its config
# ------------------------------------------------------------------------------
add_library(DBoW2 SHARED IMPORTED GLOBAL)
set_property(TARGET DBoW2 APPEND PROPERTY IMPORTED_LOCATION
  ${DBoW2_LIBRARY})
set_property(TARGET DBoW2 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${DBoW2_INCLUDE_DIR})
