set (HEADERS
    storage.h
    tensor.h
    math.h
    dimapply.h
    dimcrossapply.h
    apply.h
    linalg.h
    diagmath.h
)

set (SOURCES
    storage.c
    tensor.c
    math.c
    linalg.c
    diagmath.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIANON_SRCS ${AIANON_SRCS} ${DIR_SRCS} PARENT_SCOPE)