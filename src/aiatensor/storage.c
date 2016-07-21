#include <aiatensor/storage.h>

#ifdef ERASED_TYPE_PRESENT

AIAStorage_ *aiastorage__(empty)(void) {
  AIAStorage_ *this = aia_alloc(sizeof(AIAStorage_));
  this->data = NULL;
  this->size = 0;
  this->refcount = 1;
  return this;
}

AIAStorage_ *aiastorage__(new)(long size) {
  AIAStorage_ *this = aia_alloc(sizeof(AIAStorage_));
  this->data = aia_alloc(sizeof(T) * size);
  this->size = size;
  this->refcount = 1;
  return this;
}

AIAStorage_ *aiastorage__(newFromData)(T *data, long size) {
  AIAStorage_ *this = aia_alloc(sizeof(AIAStorage_));
  this->data = data;
  this->size = size;
  this->refcount = 1;
  return this;
}

void aiastorage__(resize)(AIAStorage_ *this, int size) {
  this->data = aia_realloc(this->data, sizeof(T) * size);
  this->size = size;
}

void aiastorage__(fill)(AIAStorage_ *this, T value) {
  int i;
  for(i = 0; i < this->size; i++)
    this->data[i] = value;
}

void aiastorage__(retain)(AIAStorage_ *this) {
  if(!this) return;
  atomic_fetch_add(&this->refcount, 1);
}

void aiastorage__(free)(AIAStorage_ *this) {
  if(!this) return;

  if(atomic_fetch_add(&this->refcount, -1) == 1) {
    free(this->data);
    free(this);
  }
}


#endif

#define ERASE_ALL
#define ERASURE_FILE "aiatensor/storage.c"
#include <aiautil/erasure.h>