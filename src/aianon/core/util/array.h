#ifndef AIA_UTIL_ARRAY_H

#include <stdlib.h>

#ifdef ERASED_TYPE_PRESENT

T *arr__(clone)(T *arr, int size);

#endif

#ifndef arr
#define arr_(type, name) AIA_FN_ERASE_(arr, type, name)
#define arr__(name) arr_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/core/util/array.h"
#include <aianon/core/erasure.h>

#define AIA_UTIL_ARRAY_H
#endif