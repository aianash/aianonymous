#include <aianon/tensor.h>

#ifdef ERASED_TYPE_AVAILABLE

T aiatensor_(T_, get)(T p) {
  return p;
}

#endif
#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor.c"
#include <aianon/util/erasure.h>