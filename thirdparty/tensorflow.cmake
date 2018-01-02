set(TENSORFLOW_VERSION 1.4.1)
set(TENSORFLOW_PREFIX tensorflow-${TENSORFLOW_VERSION})
set(TENSORFLOW_URL https://github.com/tensorflow/tensorflow/archive/v${TENSORFLOW_VERSION}.tar.gz)
set(TENSORFLOW_URL_MD5 48f80fd5ee1116f24d6f943308620a7c)

ExternalProject_Add(${TENSORFLOW_PREFIX}
    # directory to store stuff in
	PREFIX ${TENSORFLOW_PREFIX}
	
	# source
	URL ${TENSORFLOW_URL}
	URL_MD5 ${TENSORFLOW_URL_MD5}
    
    # download settings
	DOWNLOAD_NAME tensorflow-${TENSORFLOW_VERSION}.tar.gz
	DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	
	# building
	CONFIGURE_COMMAND . ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.sh && ./configure
	BUILD_COMMAND bazel build --cxxopt=-I/opt/cuda/include/crt --config=opt --config=cuda //tensorflow:libtensorflow.so
	BUILD_IN_SOURCE 1
	
	# installing
	INSTALL_COMMAND ""
)

ExternalProject_Get_Property(${TENSORFLOW_PREFIX} SOURCE_DIR)
add_library(tensorflow SHARED IMPORTED GLOBAL)
set_target_properties(tensorflow PROPERTIES IMPORTED_LOCATION ${SOURCE_DIR}/bazel-bin/tensorflow/libtensorflow.so)
set_target_properties(tensorflow PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR})
