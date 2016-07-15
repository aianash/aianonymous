cmake_minimum_required (VERSION 2.8.5 FATAL_ERROR)
cmake_policy (VERSION 2.8.5)
project (aianonymous)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

###############################################################################
# Set build features
set(CMAKE_BUILD_TYPE Debug)

###############################################################################
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CheckSymbolExists)
include(CheckTypeSize)

###############################################################################
# Check headers
set(INCLUDES "")
macro(ck_check_include_file header var)
  check_include_files("${INCLUDES};${header}" ${var})
  if(${var})
    set(INCLUDES ${INCLUDES} ${header})
  endif(${var})
endmacro(ck_check_include_file)

ck_check_include_file("stdlib.h" HAVE_STDLIB_H)

###############################################################################
# Check for integer types
# (The following are used in check.h. Regardless if they are used in
# the project, they will need to be checked in order to use Check).
check_type_size(intmax_t INTMAX_T)
check_type_size(uintmax_t UINTMAX_T)

check_type_size(pid_t PID_T)
if(NOT HAVE_PID_T)
  if(WIN32)
    set(pid_t "int")
  else(WIN32)
    MESSAGE(FATAL_ERROR "pid_t doesn't exist on this platform?")
  endif(WIN32)
endif(NOT HAVE_PID_T)

include_directories (BEFORE "${CMAKE_BINARY_DIR}")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/src")

add_subdirectory (src/aianon)
add_subdirectory (tests)

set (DEPS ${DEPS} aianon)

add_executable (aianonymous aianonymous.c)
target_link_libraries (aianonymous ${DEPS})

enable_testing ()
add_test (NAME check_aianon-math COMMAND check_aianon-math)

# {TODO] not installing executables righ now
# set(AIANONYMOUS_INSTALL_BIN_SUBDIR "bin" CACHE PATH "AIANONYMOUS install binary subdirectory")
# install (TARGETS aianonymous
#     DESTINATION "${AIANONYMOUS_INSTALL_BIN_SUBDIR}")