#include <aianon/tensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

/** accessor methods **/

//
AIAStorage_ *aiatensor__(storage)(const AIATensor_ *this) {
  return this->storage;
}

//
long aiatensor__(storageOffset)(const AIATensor_ *this) {
  return this->storageOffset;
}

//
int aiatensor__(nDimension)(const AIATensor_ *this) {
  return this->nDimension;
}

//
long aiatensor__(size)(const AIATensor_ *this, int dim) {
  return this->size[dim];
}

//
long aiatensor__(stride)(const AIATensor_ *this, int dim) {
  return this->stride[dim];
}

//
T *aiatensor__(data)(const AIATensor_ *this) {
  if(this->storage) return this->storage->data + this->storageOffset;
  else return NULL;
}

/** -- Private Helper Functions **/

//
static void aiatensor__(resize_)(AIATensor_ *this, int nDimension, long *size, long *stride);

/** creation method **/

//
AIATensor_ *aiatensor__(empty)(void) {
  AIATensor_ *this = malloc(sizeof(AIATensor_));
  RAW_TENSOR_INIT(this, 1, NULL, 0, NULL, NULL, 0);
  return this;
}

//
AIATensor_ *aiatensor__(new)(AIATensor_ *other) {
  AIATensor_ *this = malloc(sizeof(AIATensor_));
  RAW_TENSOR_INIT(this, 1, other->storage, other->storageOffset, other->size, other->stride, other->nDimension);
  return this;
}

//
AIATensor_ *aiatensor__(newVector)(int size) {
  AIATensor_ *this = aiatensor__(empty)();
  long size_[1] = {size};
  aiatensor__(resize_)(this, 1, size_, NULL);
  return this;
}

//
void aiatensor__(copy)(AIATensor_ *to, AIATensor_ *from) {
  AIA_TENSOR_APPLY2(T, to, T, from, *to_data = (T)(*from_data);)
}

//
AIATensor_ *aiatensor__(clone)(AIATensor_ *this) {
  AIATensor_ *clone = aiatensor__(empty)();
  aiatensor__(resizeAs)(clone, this);
  aiatensor__(copy)(clone, this);
  return clone;
}

//
AIATensor_ *aiatensor__(contiguous)(AIATensor_ *this) {
  if(!aiatensor__(isContiguous)(this)) return aiatensor__(clone)(this);
  else {
    aiatensor__(retain)(this);
    return this;
  }
}

//
void aiatensor__(resize)(AIATensor_ *this, TensorShape shape) {
  aiatensor__(resize_)(this, shape.nDimension, shape.size, shape.stride);
}

//
void aiatensor__(resizeAs)(AIATensor_ *this, AIATensor_ *other) {
  if(!aiatensor__(isSameSizeAs)(this, other))
    aiatensor__(resize_)(this, other->nDimension, other->size, NULL);
}

//
void aiatensor__(resize1d)(AIATensor_ *this, long size0) {
  aiatensor__(resize4d)(this, size0, -1, -1, -1);
}

//
void aiatensor__(resize2d)(AIATensor_ *this, long size0, long size1) {
  aiatensor__(resize4d)(this, size0, size1, -1, -1);
}

//
void aiatensor__(resize3d)(AIATensor_ *this, long size0, long size1, long size2) {
  aiatensor__(resize4d)(this, size0, size1, size2, -1);
}

//
void aiatensor__(resize4d)(AIATensor_ *this, long size0, long size1, long size2, long size3) {
  long size[4] = {size0, size1, size2, size3};
  aiatensor__(resize_)(this, 4, size, NULL);
}

//
void aiatensor__(set)(AIATensor_ *this, AIATensor_ *to) {
  if(this->storage != to->storage) {
    if(this->storage) aiastorage__(free)(this->storage);

    if(to->storage) {
      aiastorage__(retain)(to->storage);
      this->storage = to->storage;
    } else {
      this->storage = NULL;
    }
  }

  this->storageOffset = to->storageOffset;

  aiatensor__(resize_)(this, to->nDimension, to->size, to->stride);
}

//
void aiatensor__(narrow)(AIATensor_ *this, AIATensor_ *from, int dim, long firstIdx, long size) {
  if(!from) from = this;

  aia_argcheck(WITHIN_RANGE(dim, 0, from->nDimension), 3, "out of range");
  aia_argcheck(WITHIN_RANGE(firstIdx, 0, from->size[dim]), 4, "out of range");
  aia_argcheck((size > 0) && (firstIdx + size <= from->size[dim]), 5, "out of range");

  aiatensor__(set)(this, from);

  if(firstIdx > 0) this->storageOffset += firstIdx * this->stride[dim];

  this->size[dim] = size;
}

//
void aiatensor__(select)(AIATensor_ *this, AIATensor_ *from, int dim, int sliceIdx) {
  if(!from) from = this;

  aia_argcheck(from->nDimension > 1, 1, "cannot select a vector");
  aia_argcheck(WITHIN_RANGE(dim, 0, from->nDimension), 3, "out of range");
  aia_argcheck(WITHIN_RANGE(sliceIdx, 0, from->size[dim]), 4, "out of range");

  aiatensor__(set)(this, from);
  aiatensor__(narrow)(this, NULL, dim, sliceIdx, 1);

  int d;
  for(d = dim; d < this->nDimension - 1; d++) {
    this->size[d] = this->size[d+1];
    this->stride[d] = this->stride[d+1];
  }
  this->nDimension--;
}

//
void aiatensor__(transpose)(AIATensor_ *this, AIATensor_ *from, int dim1, int dim2) {
  aia_argcheck(WITHIN_RANGE(dim1, 0, from->nDimension), 3, "out of range");
  aia_argcheck(WITHIN_RANGE(dim2, 0, from->nDimension), 4, "out of range");

  if(from) aiatensor__(set)(this, from);

  if(dim1 == dim2) return;

  SWAP(long, this->stride[dim1], this->stride[dim2]);
  SWAP(long, this->size[dim1], this->stride[dim2]);
}

//
int aiatensor__(isSetTo)(const AIATensor_ *this, const AIATensor_ *that) {
  if(!this->storage) return 0;
  if(this->storage == that->storage &&
     this->storageOffset == that->storageOffset &&
     this->nDimension == that->nDimension) {
    int d;
    for(d = 0; d < this->nDimension; d++) {
      if(this->size[d] != that->size[d] || this->stride[d] != that->size[d])
        return 0;
    }
    return 1;
  }
  return 0;
}

//
int aiatensor__(isContiguous)(const AIATensor_ *this) {
  long z = 1;
  int d;
  for(d = this->nDimension - 1; d >= 0; d--) {
    if(this->size[d] != 1) {
      if(this->stride[d] == z)
        z *= this->size[d];
      else return 0;
    }
  }
  return 1;
}

//
int aiatensor__(isSameSizeAs)(const AIATensor_ *this, const AIATensor_ *other) {
  int d;
  if (this->nDimension != other->nDimension)
    return 0;
  for(d = 0; d < this->nDimension; ++d) {
    if(this->size[d] != other->size[d])
      return 0;
  }
  return 1;
}

//
int aiatensor__(nElement)(const AIATensor_ *this) {
  if(this->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < this->nDimension; d++)
      nElement *= this->size[d];
    return nElement;
  }
}

//
void aiatensor__(retain)(AIATensor(T_) *this) {
  if(!this) return;
  atomic_fetch_add(&this->refcount, 1);
}

//
void aiatensor__(free)(AIATensor_ *this) {
  if(!this) return;

  if(atomic_fetch_add(&this->refcount, -1)) {
    free(this->size);
    free(this->stride);
    if(this->storage) aiastorage__(free)(this->storage);
    free(this);
  }
}

//
void aiatensor__(freeCopyTo)(AIATensor_ *this, AIATensor_ *to) {
  if(this != to) aiatensor__(copy)(to, this);
  aiatensor__(free)(this);
}

/** -- Private Helper Functions **/

//
static void aiatensor__(resize_)(AIATensor_ *this, int nDimension, long *size, long *stride) {
  int d;
  long totalSize;
  int shapeAlrdyCorrect = 1;

  int nDim_ = 0;
  for(d = 0; d < nDimension; d++) {
    if(size[d] > 0) {
      nDim_ ++;

      if((this->nDimension > d) && (size[d] != this->size[d]))
        shapeAlrdyCorrect = 0;

      if((this->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != this->stride[d]))
        shapeAlrdyCorrect = 0;
    } else break;
  }

  nDimension = nDim_;

  if(nDimension != this->nDimension)
    shapeAlrdyCorrect = 0;

  if(shapeAlrdyCorrect) return;

  if(nDimension > 0) {
    if(nDimension != this->nDimension) {
      this->size = aia_realloc(this->size, sizeof(long) * nDimension);
      this->stride = aia_realloc(this->stride, sizeof(long) * nDimension);
      this->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = this->nDimension - 1; d >= 0; d--) {
      this->size[d] = size[d];
      if(stride && (stride[d] >= 0))
        this->stride[d] = stride[d];
      else {
        if(d == this->nDimension - 1)
          this->stride[d] = 1;
        else
          this->stride[d] = this->size[d+1] * this->stride[d+1];
      }
      totalSize += (this->size[d] - 1) * this->stride[d];
    }

    if(totalSize + this->storageOffset > 0) {
      if(!this->storage)
        this->storage = aiastorage__(empty)();
      if(totalSize + this->storageOffset > this->storage->size)
        aiastorage__(resize)(this->storage, totalSize + this->storageOffset);
    }
  } else {
    this->nDimension = 0;
  }
}

#endif
#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/tensor.c"
#include <aianon/core/erasure.h>