set (HEADERS
    tensor.h
)

set (SOURCES
    tensor.c
)

macro(add_curr_directory_name DIR_SRCS _srcs)
    foreach (file ${_srcs})
        set(${DIR_SRCS} ${${DIR_SRCS}} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    endforeach()
endmacro(add_curr_directory_name)

set (DIR_SRCS)

add_curr_directory_name (DIR_SRCS ${HEADERS})
add_curr_directory_name (DIR_SRCS ${SOURCES})
set (AIANON_SRCS ${AIANON_SRCS} ${DIR_SRCS} PARENT_SCOPE)