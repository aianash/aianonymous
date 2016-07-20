#include <aianon/core/util/array.h>

#ifdef ERASED_TYPE_PRESENT

T *arr__(clone)(T *arr, int size) {
  T *res = malloc(sizeof(T) * size);
  int i;
  for(i = 0; i < size; i++) res[i] = arr[i];
  return res;
}

void arr__(fill)(T *arr, const T c, const long n) {
  long i = 0;
  for(; i < n - 4; i += 4) {
    arr[i] = c;
    arr[i + 1] = c;
    arr[i + 2] = c;
    arr[i + 3] = c;
  }
  for(; i < n; i++) arr[i] = c;
}

void arr__(zero)(T *arr, const long n) {
  memset(arr, 0, n);
}

T *arr__(new)(long size) {
  T *res = malloc(sizeof(T) * size);
  memset(res, 0, size);
  return res;
}

#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/core/util/array.c"
#include <aianon/core/erasure.h>