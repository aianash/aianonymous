set (HEADERS
)

set (SOURCES
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
# add_curr_directory_name (DIR_SRCS ${SOURCES})

set (SUBMODULES kernel gp)

foreach (module ${SUBMODULES})
    add_subdirectory (${module})
endforeach()

set (AIANON_SRCS ${AIANON_SRCS} ${DIR_SRCS} PARENT_SCOPE)