#ifndef AIA_STORAGE_H

#ifdef ERASED_TYPE_AVAILABLE

typedef struct AIAStorage(T_) {
  T *data;
  long size;
  int refcount;
  char flag;
} AIAStorage(T_);

#endif

#ifndef AIAStorage
#define AIAStorage(type) AIA_STRUCT_ERASE_(type, storage)
#define aiastorage_(type, name) AIA_FN_ERASE_(storage, type, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/storage.h"
#include <aianon/util/erasure.h>

#define AIA_STORAGE_H
#endif