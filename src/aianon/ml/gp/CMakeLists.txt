set (HEADERS
    gp.h
)

set (SOURCES
    gp.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIANON_SRCS ${AIANON_SRCS} ${DIR_SRCS} PARENT_SCOPE)