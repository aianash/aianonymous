include_directories (..) # for #include <aianon/[...]>
include_directories (${CMAKE_CURRENT_BINARY_DIR}/..) # for #include <aianon/aianon_export.h>

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

macro(add_curr_directory_name DIR_SRCS _srcs)
  foreach (file ${_srcs})
    set(${DIR_SRCS} ${${DIR_SRCS}} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  endforeach()
endmacro(add_curr_directory_name)

# Define the files we need to compile
# Anything not in this list will not be compiled into aianon.
set(SOURCES
  aianon.h
  storage.h
  tensor.h
  tensor.c
)

set (AIANON_SRCS)
add_curr_directory_name (AIANON_SRCS "${SOURCES}")

# Add sub modules to this library
set (SUBMODULES core util math)

# AIANON_SRCS will be set by the subdirectories
foreach (module ${SUBMODULES})
  add_subdirectory (${module})
endforeach ()

if (BUILD_TESTS)
  add_subdirectory (tests)
endif (BUILD_TESTS)

# TODO:
# MSVC
# OpenMP
# ARM specific flags
# unix
# SSE
# AVX

# add all sources to this library
add_library (aianon SHARED ${AIANON_SRCS})
if (BUILD_STATIC)
  add_library (aianon_static STATIC ${AIANON_SRCS})
endif ()

# generate export headers
# ?? - (explain what are export headers)
include (GenerateExportHeader)
generate_export_header (aianon EXPORT_FILE_NAME aianon_export.h)
if (NOT BUILD_SHARED_LIBS)
  add_definitions (-DAIANON_STATIC_DEFINE)
endif ()

# TODO
# synchronization primitives

find_package (BLAS)
if (BLAS_FOUND)
  set (USE_BLAS 1)
  target_link_libraries (aianon ${BLAS_LIBRARIES})
endif (BLAS_FOUND)

find_package (LAPACK)
if (LAPACK_FOUND)
  set (USE_LAPACK 1)
  target_link_libraries (aianon ${LAPACK_LIBRARIES})
endif (LAPACK_FOUND)

# check for inline and __thread support

# create config file where settings from cmake is inserted
configure_file ("config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/aianon/config.h")
set(AIANON_SRCS ${AIANON_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/aianon/config.h")

set_target_properties (aianon
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

# set (INCLUDE_FILES ${INCLUDE_FILES} "${CMAKE_CURRENT_BINARY_DIR}/aianon/config.h")

add_custom_target(aianon_headers)
add_custom_command(TARGET aianon_headers POST_BUILD
  COMMENT "Moving header files to include/aianon"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aianon/
  COMMAND ${CMAKE_COMMAND} ARGS -E
    copy ${CMAKE_CURRENT_BINARY_DIR}/aianon_export.h
         ${CMAKE_BINARY_DIR}/include/aianon/
  COMMAND ${CMAKE_COMMAND} ARGS -E
    copy ${CMAKE_CURRENT_BINARY_DIR}/aianon/config.h
         ${CMAKE_BINARY_DIR}/include/aianon/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aianon_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aianon/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aianon DESTINATION include)

install (TARGETS aianon
  EXPORT aianon-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# # Create AIASettings.cmake
# get_target_property(AIANON_OUTPUT_NAME AIANON LOCATION)
# get_filename_component(AIANON_OUTPUT_NAME ${AIANON_OUTPUT_NAME} NAME)
# set(AIANON_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_LIB_SUBDIR}/${AIANON_OUTPUT_NAME}")
# set(AIANON_INCLUDE_DIR $"{CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_INCLUDE_SUBDIR}/aianon")
# CONFIGURE_FILE(AIASettings.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake")
# INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake"
#   DESTINATION "${AIANON_INSTALL_CMAKE_SUBDIR}")

add_dependencies(aianon aianon_headers)