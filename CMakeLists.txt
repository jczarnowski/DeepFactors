cmake_minimum_required(VERSION 3.10)

project(nslam)

add_subdirectory(thirdparty)

add_executable(tftest tftest.cpp)
target_link_libraries(tftest PUBLIC tensorflow)
add_dependencies(tftest tensorflow-1.4.1)
