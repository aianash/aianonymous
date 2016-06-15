cmake_minimum_required (VERSION 2.8.5 FATAL_ERROR)
cmake_policy (VERSION 2.8.5)
project (aianonymous)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# include(AIAnonymousPaths)
# include(AIAnonymousPathsInit)
# include(AIAnonymousPackage)

include_directories (BEFORE "${CMAKE_BINARY_DIR}")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/src/aianon")

add_subdirectory (src/aianon)

set (DEPS ${DEPS} aianon)
# include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/src/aianon")


# set(src
#     Test.c)

add_executable (aianonymous aianonymous.c)
target_link_libraries (aianonymous ${DEPS})

set(AIANONYMOUS_INSTALL_BIN_SUBDIR "bin" CACHE PATH "AIANONYMOUS install binary subdirectory")

install (TARGETS aianonymous
    DESTINATION "${AIANONYMOUS_INSTALL_BIN_SUBDIR}")