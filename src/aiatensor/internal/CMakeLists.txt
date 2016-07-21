set (HEADERS
    lapack.h
    blas.h

)

set (SOURCES
    lapack.c
    blas.c
)

set (DIR_SRCS)
add_curr_directory_name (DIR_SRCS "${HEADERS}")
add_curr_directory_name (DIR_SRCS "${SOURCES}")

set (AIATENSOR_SRCS ${AIATENSOR_SRCS} ${DIR_SRCS} PARENT_SCOPE)