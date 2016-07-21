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
    gp.h
)

set (SOURCES
    gp.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIAGP_SRCS ${AIAGP_SRCS} ${DIR_SRCS})

# add all sources to this library
add_library (aiagp SHARED ${AIAGP_SRCS})
if (BUILD_STATIC)
  add_library (aiagp_static STATIC ${AIAGP_SRCS})
endif ()

ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)

set_target_properties (aiagp
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_custom_target(aiagp_headers)
add_custom_command(TARGET aiagp_headers POST_BUILD
  COMMENT "Moving header files to include/aianon"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aiagp/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aiagp_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aiagp/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aiagp DESTINATION include)

install (TARGETS aiagp
  EXPORT aiagp-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

add_dependencies(aiagp aiagp_headers)