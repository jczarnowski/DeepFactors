add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(cuda)

if(DF_BUILD_DEMO)
  add_subdirectory(demo)
endif()

if(DF_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(DF_BUILD_TOOLS OR DF_BUILD_DEMO)
  add_subdirectory(drivers)
  add_subdirectory(gui)
endif()


