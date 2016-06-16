#include <aianon/tensor.h>

#ifdef ERASURE_ACTIVE

T AIAErase_(get)(T p) {
  return p;
}

#endif
#define ERASE_BYTE
#define ERASE_CHAR
#define ERASE_SHORT
#define ERASE_INT
#define ERASE_LONG
#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/tensor.c"
#include <aianon/util/erasure.h>