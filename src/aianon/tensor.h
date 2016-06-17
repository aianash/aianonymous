#ifndef AIA_TENSOR_H

#include <aianon/storage.h>

#ifdef ERASED_TYPE_AVAILABLE

typedef struct AIATensor(T_) {
  long *size;
  long *stride;
  int nDimension;

  AIAStorage(T_) *storage;
  long storageOffset;
  int refcount;

  char flag;

} AIATensor(T_);

AIATensor(T_) *aiatensor_(T_, new)(void);

#endif

#ifndef AIATensor
#define AIATensor(type) AIA_STRUCT_ERASE_(type, tensor)
#define aiatensor_(type, name) AIA_FN_ERASE_(tensor, type, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor.h"
#include <aianon/util/erasure.h>

#define AIA_TENSOR_H
#endif