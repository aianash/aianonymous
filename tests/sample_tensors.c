#include "sample_tensors.h"

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

long size4x4[2] = {4L, 4L};
TensorShape shape4x4 = NEW_TENSOR_SHAPE(2, size4x4, NULL);

long size3x3[2] = {3L, 3L};
TensorShape shape3x3 = NEW_TENSOR_SHAPE(2, size3x3, NULL);

#endif

#ifdef ERASED_TYPE_PRESENT

static T *clone(T *arr, int size) {
  T *res = aia_alloc(sizeof(T) * size);
  int i;
  for(i = 0; i < size; i++) res[i] = arr[i];
  return res;
}

T Mat_(rnd4x4)[16] =
  { 0.44783241,  0.06268606,  0.03410412,  0.65443608,
    0.96924508,  0.85259425,  0.08037111,  0.6618764 ,
    0.93053844,  0.21757096,  0.5980586 ,  0.66421652,
    0.95874301,  0.55492546,  0.87603126,  0.8948417  };

T Mat_(pd3x3)[9] =
  { 0.29291524,  0.06772514,  0.00308962,
    0.06772514,  0.55844815,  0.02191619,
    0.00308962,  0.02191619,  0.02722591 };


AIATensor_ *mktnsr__(rnd4x4)(void) {
  return aiatensor__(newFromData)(clone(Mat_(rnd4x4), 16), shape4x4);
}

AIATensor_ *mktnsr__(pd3x3)(void) {
  return aiatensor__(newFromData)(clone(Mat_(pd3x3), 9), shape3x3);
}


#endif

#define ERASE_FLOAT
#define ERASURE_FILE "sample_tensors.c"
#include <aianon/core/erasure.h>