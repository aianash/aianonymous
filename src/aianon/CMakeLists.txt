cmake_minimum_required(VERSION 2.8.5)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# When compiled standalone then don't inherit this property from
# parent directory's CMakeLists.txt
if(NOT AIANON_INSTALL_BIN_SUBDIR
  OR NOT AIANON_INSTALL_LIB_SUBDIR
  OR NOT AIANON_INSTALL_INCLUDE_SUBDIR
  OR NOT AIANON_INSTALL_CMAKE_SUBDIR)

  set(AIANON_INSTALL_BIN_SUBDIR "bin" CACHE PATH "AIANON install binary subdirectory")
  set(AIANON_INSTALL_LIB_SUBDIR "lib" CACHE PATH "AIANON install library subdirectory")
  set(AIANON_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "AIANON install include subdirectory")
  set(AIANON_INSTALL_CMAKE_SUBDIR "share/cmake/aianon" CACHE PATH "AIANON install cmake subdirectory")
endif()

# TODO:
# MSVC
# OpenMP
# ARM specific flags
# unix
# SSE
# AVX

set(headers
  AIAConfig.h
  AIATensor.h
  )

set(sources
  AIATensor.c
  )

set(src ${sources} ${headers})

add_library(aianon SHARED ${src})
if(BUILD_STATIC)
  add_library(aianon_static STATIC ${src})
endif()

# TODO
# synchronization primitives

find_package(BLAS)
if(BLAS_FOUND)
  set(USE_BLAS 1)
  TARGET_LINK_LIBRARIES(aianon ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

find_package(LAPACK)
if(LAPACK_FOUND)
  set(USE_LAPACK 1)
  TARGET_LINK_LIBRARIES(aianon ${LAPACK_LIBRARIES})
endif(LAPACK_FOUND)

# check for inline and __thread support

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE("AIAConfig.h.in" "${CMAKE_CURRENT_BINARY_DIR}/AIAConfig.h")

install(TARGETS aianon
  EXPORT aianon-exports
  RUNTIME DESTINATION "${AIANON_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${AIANON_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${AIANON_INSTALL_LIB_SUBDIR}")

install(FILES
  AIATensor.h
  ${CMAKE_CURRENT_BINARY_DIR}/AIAConfig.h
  DESTINATION "${AIANON_INSTALL_INCLUDE_SUBDIR}/aianon")

# Create AIASettings.cmake
get_target_property(AIANON_OUTPUT_NAME AIANON LOCATION)
get_filename_component(AIANON_OUTPUT_NAME ${AIANON_OUTPUT_NAME} NAME)
set(AIANON_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_LIB_SUBDIR}/${AIANON_OUTPUT_NAME}")
set(AIANON_INCLUDE_DIR $"{CMAKE_INSTALL_PREFIX}/${AIANON_INSTALL_INCLUDE_SUBDIR}/aianon")
CONFIGURE_FILE(AIASettings.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/AIASettings.cmake"
  DESTINATION "${AIANON_INSTALL_CMAKE_SUBDIR}")