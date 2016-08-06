include_directories (..) # for #include <aiatensor/[...]>
include_directories (${CMAKE_CURRENT_BINARY_DIR}/..) # for #include <aiatensor/aiatensor_export.h>

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

macro(add_curr_directory_name DIR_SRCS _srcs)
  foreach (file ${_srcs})
    set(${DIR_SRCS} ${${DIR_SRCS}} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  endforeach()
endmacro(add_curr_directory_name)

# Define the files we need to compile
# Anything not in this list will not be compiled into aiatensor.
set(HEADERS
    aianon.h
    storage.h
    tensor.h
    math.h
    dimapply.h
    dimcrossapply.h
    apply.h
    linalg.h
    diagmath.h
    blas.h
    lapack.h
    types.h
    functional.h
)

set(SOURCES
    storage.c
    tensor.c
    math.c
    linalg.c
    diagmath.c
    blas.c
    lapack.c
)

set (AIATENSOR_SRCS)
add_curr_directory_name (AIATENSOR_SRCS "${HEADERS}")
add_curr_directory_name (AIATENSOR_SRCS "${SOURCES}")

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
add_library (aiatensor SHARED ${AIATENSOR_SRCS})
if (BUILD_STATIC)
  add_library (aiatensor_static STATIC ${AIATENSOR_SRCS})
endif ()

# generate export headers
# ?? - (explain what are export headers)
include (GenerateExportHeader)
generate_export_header (aiatensor EXPORT_FILE_NAME aiatensor_export.h)
if (NOT BUILD_SHARED_LIBS)
  add_definitions (-DAIANON_STATIC_DEFINE)
endif ()

# TODO
# synchronization primitives
# based on compiler
# currently defaulted to stc=c11
ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)

find_package (BLAS)
if (BLAS_FOUND)
  set (USE_BLAS 1)
  target_link_libraries (aiatensor ${BLAS_LIBRARIES})
endif (BLAS_FOUND)

find_package (LAPACK)
if (LAPACK_FOUND)
  set (USE_LAPACK 1)
  target_link_libraries (aiatensor ${LAPACK_LIBRARIES})
endif (LAPACK_FOUND)

# check for inline and __thread support

# # create config file where settings from cmake is inserted
# configure_file ("config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/aiatensor/config.h")
# set(AIATENSOR_SRCS ${AIATENSOR_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/aiatensor/config.h")

set_target_properties (aiatensor
  PROPERTIES
  VERSION 1.0  # build version
  SOVERSION 1  # API version
)

file (GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

# set (INCLUDE_FILES ${INCLUDE_FILES} "${CMAKE_CURRENT_BINARY_DIR}/aiatensor/config.h")

add_custom_target(aiatensor_headers)
add_custom_command(TARGET aiatensor_headers POST_BUILD
  COMMENT "Moving header files to include/aiatensor"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/aiatensor/
  COMMAND ${CMAKE_COMMAND} ARGS -E
    copy ${CMAKE_CURRENT_BINARY_DIR}/aiatensor_export.h
         ${CMAKE_BINARY_DIR}/include/aiatensor/
  # COMMAND ${CMAKE_COMMAND} ARGS -E
  #   copy ${CMAKE_CURRENT_BINARY_DIR}/aiatensor/config.h
  #        ${CMAKE_BINARY_DIR}/include/aiatensor/
)

foreach(inc_file ${INCLUDE_FILES})
  add_custom_command(TARGET aiatensor_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
           ${CMAKE_BINARY_DIR}/include/aiatensor/${inc_file})
endforeach()

install (DIRECTORY ${CMAKE_BINARY_DIR}/include/aiatensor DESTINATION include)

install (TARGETS aiatensor
  EXPORT aiatensor-exports
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# # Create AIASettings.cmake
# get_target_property(AIANON_OUTPUT_NAME AIANON LOCATION)
# get_filename_component(AIANON_OUTPUT_NAME ${AIANON_OUTPUT_NAME} NAME)
# set(AIANON_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_LIB_SUBDIR}/${AIANON_OUTPUT_NAME}")
# set(AIANON_INCLUDE_DIR $"{CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_INCLUDE_SUBDIR}/aiatensor")
# CONFIGURE_FILE(AIASettings.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake")
# INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake"
#   DESTINATION "${AIANON_INSTALL_CMAKE_SUBDIR}")

add_dependencies(aiatensor aiatensor_headers)