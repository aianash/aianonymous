#ifndef AIA_TENSOR_H

#include <stdatomic.h>
#include <aianon/core/util.h>
#include <aianon/tensor/storage.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

typedef struct TensorShape {
  long *size;
  long *stride;
  int nDimension;
} TensorShape;

#define NEW_TENSOR_SHAPE(nDimension_, size_, stride_) \
{ .nDimension=(nDimension_), .size=(size_), .stride=(stride_) }

#define RAW_TENSOR_INIT(this, refcount_, storage_, offset_, size_, stride_, nDimension_) \
{                                                                                 \
  this->refcount = refcount_;                                                     \
  this->storage = storage_;                                                       \
  this->storageOffset = offset_;                                                  \
  this->size = size_;                                                             \
  this->stride = stride_;                                                         \
  this->nDimension = nDimension_;                                                 \
}


#endif

#ifdef ERASED_TYPE_PRESENT

typedef struct AIATensor_ {
  long *size;
  long *stride;
  int nDimension;

  struct AIAStorage_ *storage;
  long storageOffset;
  int refcount;

} AIATensor_;

/** accessor methods **/
AIA_API AIAStorage_ *aiatensor__(storage)(const AIATensor_ *this);
AIA_API long aiatensor__(storageOffset)(const AIATensor_ *this);
AIA_API int aiatensor__(nDimension)(const AIATensor_ *this);
AIA_API long aiatensor__(size)(const AIATensor_ *this, int dim);
AIA_API long aiatensor__(stride)(const AIATensor_ *this, int dim);
AIA_API T *aiatensor__(data)(const AIATensor_ *this);

/** creation method **/
AIA_API AIATensor_ *aiatensor__(empty)(void);
AIA_API AIATensor_ *aiatensor__(newFromTensor)(AIATensor_ *other);

AIA_API AIATensor_ *aiatensor__(clone)(AIATensor_ *this);

AIA_API AIATensor_ *aiatensor__(contiguous)(AIATensor_ *this);
AIA_API AIATensor_ *aiatensor__(newTransposed)(AIATensor_ *this, int dim1_, int dim2_);

AIA_API void aiatensor__(resize)(AIATensor_ *this, TensorShape shape);
AIA_API void aiatensor__(resizeAs)(AIATensor_ *this, AIATensor_ *other);

AIA_API void aiatensor__(replace)(AIATensor_ *this, AIATensor_ *other);

// AIA_API void aiatensor__(set)(AIATensor_ *this, AIATensor_ src);

AIA_API void aiatensor__(transpose)(AIATensor_ *this, int dim1_, int dim2_);
AIA_API void aiatensor__(transposeFrom)(AIATensor_ *this, AIATensor_ *from, int dim1_, int dim2_);

AIA_API int aiatensor__(isContiguous)(const AIATensor_ *this);
AIA_API int aiatensor__(isSameSizeAs)(const AIATensor_ *this, const AIATensor_ *other);
AIA_API int aiatensor__(nElement)(const AIATensor_ *this);


AIA_API void aiatensor__(retain)(AIATensor_ *this);
AIA_API void aiatensor__(free)(AIATensor_ *this);

#endif

#ifndef AIATensor
// explicits
#define AIATensor(type) AIA_STRUCT_ERASE_(type, tensor)
#define aiatensor_(type, name) AIA_FN_ERASE_(tensor, type, name)

// implicitly typed
#define AIATensor_ AIATensor(T_)
#define aiatensor__(name) aiatensor_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/tensor.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_H
#endif