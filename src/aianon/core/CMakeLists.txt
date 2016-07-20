set (HEADERS
    erasure.h
)

set (SOURCES
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
# add_curr_directory_name (DIR_SRCS ${SOURCES})

set (SUBMODULES math util)

foreach (module ${SUBMODULES})
    add_subdirectory (${module})
endforeach()

set (AIANON_SRCS ${AIANON_SRCS} ${DIR_SRCS} PARENT_SCOPE)