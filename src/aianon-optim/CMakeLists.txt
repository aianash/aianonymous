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
    optim.h
)

set (SOURCES
    sgd.c
    adagrad.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIANON_OPTIM_SRCS ${AIANON_OPTIM_SRCS} ${DIR_SRCS})

# add all sources to this library
add_library (aianon-optim SHARED ${AIANON_OPTIM_SRCS})
if (BUILD_STATIC)
  add_library (aianon_static STATIC ${AIANON_OPTIM_SRCS})
endif ()

ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)

set_target_properties (aianon-optim
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_custom_target(aianon-optim_headers)
add_custom_command(TARGET aianon-optim_headers POST_BUILD
  COMMENT "Moving header files to include/aianon"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aianon-optim/
  # COMMAND ${CMAKE_COMMAND} ARGS -E
  #   copy ${CMAKE_CURRENT_BINARY_DIR}/aianon/config.h
  #        ${CMAKE_BINARY_DIR}/include/aianon/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aianon-optim_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aianon-optim/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aianon-optim DESTINATION include)

install (TARGETS aianon-optim
  EXPORT aianon-optim-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

set (DEPS ${DEPS} aianon)
target_link_libraries (aianon-optim ${DEPS})

add_dependencies(aianon-optim aianon-optim_headers)
