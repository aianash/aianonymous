include_directories (..)
include_directories (${CMAKE_CURRENT_BINARY_DIR}/..)

macro(add_curr_directory_name DIR_SRCS _srcs)
  foreach (file ${_srcs})
    set(${DIR_SRCS} ${${DIR_SRCS}} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  endforeach()
endmacro(add_curr_directory_name)

# Define the files we need to compile
# Anything not in this list will not be compiled into aianon.

set (HEADERS
    kernel.h
)

set (SOURCES
    kernel.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIAKERNEL_SRCS ${AIAKERNEL_SRCS} ${DIR_SRCS})

# add all sources to this library
add_library (aiakernel SHARED ${AIAKERNEL_SRCS})
if (BUILD_STATIC)
  add_library (aianon_static STATIC ${AIAKERNEL_SRCS})
endif ()

ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)

set_target_properties (aiakernel
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_custom_target(aiakernel_headers)
add_custom_command(TARGET aiakernel_headers POST_BUILD
  COMMENT "Moving header files to include/aianon"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aiakernel/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aiakernel_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aiakernel/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aiakernel DESTINATION include)

install (TARGETS aiakernel
  EXPORT aiakernel-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

add_dependencies(aiakernel aiakernel_headers)