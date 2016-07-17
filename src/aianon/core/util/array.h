#ifndef AIA_UTIL_ARRAY_H

#include <stdlib.h>
#include <memory.h>

#ifdef ERASED_TYPE_PRESENT

T *arr__(clone)(T *arr, int size);
void arr__(fill)(T *arr, const T c, const long n);
void arr__(zero)(T *arr, const long n);

#endif

#ifndef arr
#define arr_(type, name) AIA_FN_ERASE_(arr, type, name)
#define arr__(name) arr_(T_, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/core/util/array.h"
#include <aianon/core/erasure.h>

#define AIA_UTIL_ARRAY_H
#endif