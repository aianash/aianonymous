#ifndef AIA_UTIL_MEMORY_H
#define AIA_UTIL_MEMORY_H

#include <stdlib.h>
#include <aianon/util/check.h>

void *aia_alloc(long size);
void *aia_realloc(void *ptr, long size);

#endif