#ifndef AIA_CORE_UTIL_ARRAY_H

#include <stdlib.h>
#include <memory.h>

#ifdef ERASED_TYPE_PRESENT

T *arr__(clone)(T *arr, int size);
void arr__(fill)(T *arr, const T c, const long n);
void arr__(zero)(T *arr, const long n);
T *arr__(new)(long size);


#endif

#ifndef arr
#define arr_(type, name) AIA_FN_ERASE_(arr, type, name)
#define arr__(name) arr_(T_, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aiautil/array.h"
#include <aiautil/erasure.h>

#define AIA_CORE_UTIL_ARRAY_H
#endif