# Find TensorFlow C API
# Author: Jan Czarnowski (jancz@tuta.io)

# find the main header
find_path(
  TENSORFLOW_INCLUDE_DIR
    tensorflow/c/c_api.h
  HINTS 
    /usr/include/tensorflow 
    /usr/local/include/tensorflow
)

# find both libraries
find_library(TENSORFLOW_LIBRARY libtensorflow.so)
find_library(TENSORFLOW_FRAMEWORK_LIBRARY libtensorflow_framework.so)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY TENSORFLOW_FRAMEWORK_LIBRARY)

# create an imported target
if(TensorFlow_FOUND)
  message(STATUS "Found TensorFlow (include: ${TENSORFLOW_INCLUDE_DIR}, library: ${TENSORFLOW_LIBRARY})")

  # provide commonly used aliases
  set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY} ${TENSORFLOW_FRAMEWORK_LIBRARY})

  # libtensorflow
  add_library(TensorFlowBase SHARED IMPORTED)
  set_property(TARGET TensorFlowBase PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${TENSORFLOW_INCLUDE_DIRS})
  set_property(TARGET TensorFlowBase PROPERTY IMPORTED_LOCATION ${TENSORFLOW_LIBRARY})

  # libtensorflow_framework
  add_library(TensorFlowFramework SHARED IMPORTED)
  set_property(TARGET TensorFlowFramework PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${TENSORFLOW_INCLUDE_DIRS})
  set_property(TARGET TensorFlowFramework PROPERTY IMPORTED_LOCATION ${TENSORFLOW_FRAMEWORK_LIBRARY})

  # aggregate the above in a single target
  add_library(TensorFlow INTERFACE IMPORTED)
  set_property(TARGET TensorFlow PROPERTY
      INTERFACE_LINK_LIBRARIES TensorFlowBase TensorFlowFramework)
endif()
