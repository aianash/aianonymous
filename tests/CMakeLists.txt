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

set(AIANON_MATH_TEST_SOURCES
    check_aianon-math_tensor.c
    check_aianon-math_tensormath.c
    check_aianon-math_tensorlinalg.c
    check_aianon-math.c
)

set(AIANON_MATH_TEST_HEADERS
    check_aianon-math.h)

add_executable (check_aianon-math ${AIANON_MATH_TEST_HEADERS} ${AIANON_MATH_TEST_SOURCES})
target_link_libraries (check_aianon-math aianon ${CHECK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})