# set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include_directories (BEFORE "${CMAKE_BINARY_DIR}")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/src")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/tests")

###############################################################################
# Generate "config.h" from "cmake/config.h.cmake"
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DHAVE_CONFIG_H)
set(CONFIG_HEADER ${CMAKE_CURRENT_BINARY_DIR}/config.h)

find_package (Threads)

find_package (Check REQUIRED)
include_directories(${CHECK_INCLUDE_DIRS})

set(AIA_TEST_SOURCES
    check_aiatensor.c
    check_aiatensor_math.c
    check_aiatensor_linalg.c
    check_aiakernel.c
    check_aiagp.c
    check_aiaoptim.c
    check_aia.c
)

set(AIA_TEST_HEADERS
    check_aia.h
)

add_executable (check_aia ${AIA_TEST_HEADERS} ${AIA_TEST_SOURCES})
target_link_libraries (check_aia aiaoptim aiagp aiakernel aiarandom aiatensor aiautil ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})