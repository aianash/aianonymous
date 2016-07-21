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
    array.h
    check.h
    erasure.h
    memory.h
    util.h
)

set (SOURCES
    array.c
    check.c
    memory.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIAUTIL_SRCS ${AIAUTIL_SRCS} ${DIR_SRCS})

# add all sources to this library
add_library (aiautil SHARED ${AIAUTIL_SRCS})
if (BUILD_STATIC)
  add_library (aiautil_static STATIC ${AIAUTIL_SRCS})
endif ()

ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)

set_target_properties (aiautil
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_custom_target(aiautil_headers)
add_custom_command(TARGET aiautil_headers POST_BUILD
  COMMENT "Moving header files to include/aiautil"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aiautil/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aiautil_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aiautil/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aiautil DESTINATION include)

install (TARGETS aiautil
  EXPORT aiautil-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

add_dependencies(aiautil aiautil_headers)
