#include <aianon/tensor.h>

#ifdef ERASED_TYPE_AVAILABLE

AIATensor(T_) *aiatensor_(T_, new)(void) {
  AIATensor(T_) *tnsr;
  return tnsr;
}

#endif
#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor.c"
#include <aianon/util/erasure.h>