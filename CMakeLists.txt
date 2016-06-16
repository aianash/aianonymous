cmake_minimum_required (VERSION 2.8.5 FATAL_ERROR)
cmake_policy (VERSION 2.8.5)
project (aianonymous)

include_directories (BEFORE "${CMAKE_BINARY_DIR}")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/src")

add_subdirectory (src/aianon)

set (DEPS ${DEPS} aianon)

add_executable (aianonymous aianonymous.c)
target_link_libraries (aianonymous ${DEPS})

set(AIANONYMOUS_INSTALL_BIN_SUBDIR "bin" CACHE PATH "AIANONYMOUS install binary subdirectory")

# not installing executables righ now
# install (TARGETS aianonymous
#     DESTINATION "${AIANONYMOUS_INSTALL_BIN_SUBDIR}")