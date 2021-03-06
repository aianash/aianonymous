#ifndef AIA_TENSOR_STORAGE_H

#include <stdatomic.h>
#include <aiautil/util.h>

#ifdef ERASED_TYPE_PRESENT

typedef struct AIAStorage_ {
  T *data;
  long size;
  int refcount;
} AIAStorage_;

AIA_API AIAStorage_ *aiastorage__(empty)(void);
AIA_API AIAStorage_ *aiastorage__(new)(long size);
AIA_API AIAStorage_ *aiastorage__(newFromData)(T *data, long size);


AIA_API void aiastorage__(retain)(AIAStorage_ *this);
AIA_API void aiastorage__(free)(AIAStorage_ *this);
AIA_API void aiastorage__(resize)(AIAStorage_ *this, int size);

#endif

#ifndef AIAStorage
// explicit
#define AIAStorage(type) AIA_STRUCT_ERASE_(type, storage)
#define aiastorage_(type, name) AIA_FN_ERASE_(storage, type, name)

// implicit
#define AIAStorage_ AIAStorage(T_)
#define aiastorage__(name) aiastorage_(T_, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aiatensor/storage.h"
#include <aiautil/erasure.h>

#define AIA_TENSOR_STORAGE_H
#endif