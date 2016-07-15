#ifndef AIA_TENSOR_H

#include <stdatomic.h>
#include <aianon/core/util.h>
#include <aianon/tensor/storage.h>
#include <aianon/tensor/apply.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

#define RAW_TENSOR_INIT(this, refcount_, storage_, offset_, size_, stride_, nDimension_) \
{                                                                                        \
  this->refcount = refcount_;                                                            \
  this->storage = storage_;                                                              \
  this->storageOffset = offset_;                                                         \
  this->size = size_;                                                                    \
  this->stride = stride_;                                                                \
  this->nDimension = nDimension_;                                                        \
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
AIA_API int aiatensor__(nElement)(const AIATensor_ *this);

/** Tests **/
//---------//

AIA_API int aiatensor__(isContiguous)(const AIATensor_ *this);
AIA_API int aiatensor__(isSameSizeAs)(const AIATensor_ *this, const AIATensor_ *other);
AIA_API int aiatensor__(isSameShape)(const AIATensor_ *this, int nDimension, long *size, long *stride);

/** Tensor constructors **/
AIA_API AIATensor_ *aiatensor__(empty)(void);
AIA_API AIATensor_ *aiatensor__(new)(AIATensor_ *other);
AIA_API AIATensor_ *aiatensor__(newVector)(int size);
AIA_API AIATensor_ *aiatensor__(newFromData)(T *data, int nDimension, long *size, long *stride);

// AIA_API AIATensor_ *aiatensor__(newWithStorage)(AIAStorage_ *storage, long storageOffset, TensorShape shape);
// AIA_API AIATensor_ *aiatensor__(newOfShape)(TensorShape shape);

/** Referencing a tensor to an existing tensor or chunk of memory **/
//-----------------------------------------------------------------//

// Set this to view the same storage as "to" tensor
AIA_API void aiatensor__(set)(AIATensor_ *this, AIATensor_ *to);
AIA_API int aiatensor__(isSetTo)(const AIATensor_ *this, const AIATensor_ *that);

/** Cloning **/
//-----------//

// Returns a clone of a tensor. The memory is copied.
AIA_API AIATensor_ *aiatensor__(clone)(AIATensor_ *this);

// If the given Tensor contents are contiguous in memory, returns the exact same Tensor (no memory copy).
// Otherwise (not contiguous in memory), returns a clone (memory copy).
AIA_API AIATensor_ *aiatensor__(contiguous)(AIATensor_ *this);

/** Resizing **/
//------------//

AIA_API void aiatensor__(resize)(AIATensor_ *this, int nDimension, long *size, long *stride);
AIA_API void aiatensor__(resizeAs)(AIATensor_ *this, AIATensor_ *other);
AIA_API void aiatensor__(resize1d)(AIATensor_ *this, long size0);
AIA_API void aiatensor__(resize2d)(AIATensor_ *this, long size0, long size1);
AIA_API void aiatensor__(resize3d)(AIATensor_ *this, long size0, long size1, long size2);
AIA_API void aiatensor__(resize4d)(AIATensor_ *this, long size0, long size1, long size2, long size3);

/** Extracting sub tensors **/
//--------------------------//
// Each of these methods returns a Tensor which is a sub-tensor of the given tensor
// These methods are very fast, as they do not involve any memory copy.

AIA_API void aiatensor__(narrow)(AIATensor_ *this, AIATensor_ *from, int dim, long firstIdx, long size);
AIA_API void aiatensor__(select)(AIATensor_ *this, AIATensor_ *from, int dim, int indx);


/** Manipulating tensor view **/
//----------------------------//
// Methods returns a Tensor which is another way of viewing the Storage of the given tensor.
// Hence, any modification in the memory of the sub-tensor will have an impact on the
// primary tensor, and vice-versa.
// [TODO]
// unfold, view, viewAs

AIA_API void aiatensor__(transpose)(AIATensor_ *this, AIATensor_ *from, int dim1_, int dim2_);

/** copying and initializing methods */
//-----------------------------------//
// TODO: fill. zeros
AIA_API void aiatensor__(copy)(AIATensor_ *to, AIATensor_ *from);

/** Expanding/Replicating/Squeezing Tensors **/
//-------------------------------------------//
// TODO: expandAs, squeeze


AIA_API void aiatensor__(retain)(AIATensor_ *this);
AIA_API void aiatensor__(free)(AIATensor_ *this);
AIA_API void aiatensor__(freeCopyTo)(AIATensor_ *this, AIATensor_ *to);

AIA_API bool aiatensor__(isVector)(AIATensor_ *a);
AIA_API bool aiatensor__(isMatrix)(AIATensor_ *a);
AIA_API bool aiatensor__(isSquare)(AIATensor_ *a);



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