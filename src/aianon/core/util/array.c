#include <aianon/core/util/array.h>

#ifdef ERASED_TYPE_PRESENT

T *arr__(clone)(T *arr, int size) {
  T *res = malloc(sizeof(T) * size);
  int i;
  for(i = 0; i < size; i++) res[i] = arr[i];
  return res;
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/core/util/array.c"
#include <aianon/core/erasure.h>