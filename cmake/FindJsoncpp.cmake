# Find jsoncpp
#
# Find the jsoncpp includes and library
# source: https://github.com/cinemast/libjson-rpc-cpp/blob/master/cmake/FindJsoncpp.cmake

# This module defines
#  JSONCPP_INCLUDE_DIR, where to find header, etc.
#  JSONCPP_LIBRARY, the libraries needed to use jsoncpp.
#  Jsoncpp_FOUND, If false, do not try to use jsoncpp.
#  JSONCPP_INCLUDE_PREFIX, include prefix for jsoncpp.
#  jsoncpp imported target
#

# only look in default directories
find_path(
	JSONCPP_INCLUDE_DIR
	NAMES json/json.h
	PATH_SUFFIXES jsoncpp
	DOC "jsoncpp include dir"
)

find_library(
    JSONCPP_LIBRARY
    NAMES jsoncpp
    DOC "jsoncpp library"
)

add_library(jsoncpp UNKNOWN IMPORTED)
set_target_properties(
	jsoncpp
	PROPERTIES
	IMPORTED_LOCATION "${JSONCPP_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIR}"
)

# find JSONCPP_INCLUDE_PREFIX
find_path(
    JSONCPP_INCLUDE_PREFIX
    NAMES json.h
    PATH_SUFFIXES jsoncpp/json json
)

if (${JSONCPP_INCLUDE_PREFIX} MATCHES "jsoncpp")
    set(JSONCPP_INCLUDE_PREFIX "jsoncpp/json")
else()
    set(JSONCPP_INCLUDE_PREFIX "json")
endif()

# handle the QUIETLY and REQUIRED arguments and set JSONCPP_FOUND to TRUE
# if all listed variables are TRUE, hide their existence from configuration view
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jsoncpp DEFAULT_MSG JSONCPP_INCLUDE_DIR JSONCPP_LIBRARY)
mark_as_advanced (JSONCPP_INCLUDE_DIR JSONCPP_LIBRARY)
