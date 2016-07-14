#include "sample_tensors.h"

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

long size4x4[2] = {4L, 4L};
TensorShape shape4x4 = NEW_TENSOR_SHAPE(2, size4x4, NULL);

long size3x3[2] = {3L, 3L};
TensorShape shape3x3 = NEW_TENSOR_SHAPE(2, size3x3, NULL);

#endif

#ifdef ERASED_TYPE_PRESENT

static T *mktnsr__(clone)(T *arr, int size) {
  T *res = aia_alloc(sizeof(T) * size);
  int i;
  for(i = 0; i < size; i++) res[i] = arr[i];
  return res;
}

T Mat_(rnd4x4)[16] =
  { 0.07991805,  0.97586422,  0.2668651 ,  0.9124779 ,
    0.84907812,  0.66531035,  0.93157663,  0.71106513,
    0.66486698,  0.30385106,  0.21018787,  0.77947325,
    0.41738024,  0.90427542,  0.16466475,  0.09119455 };

T Mat_(rnd4x4T)[16] =
  { 0.07991805,  0.84907812,  0.66486698,  0.41738024,
    0.97586422,  0.66531035,  0.30385106,  0.90427542,
    0.2668651 ,  0.93157663,  0.21018787,  0.16466475,
    0.9124779 ,  0.71106513,  0.77947325,  0.09119455 };

T Mat_(rnd3x3)[9] =
  { 0.07991805,  0.97586422,  0.2668651 ,
    0.84907812,  0.66531035,  0.93157663,
    0.66486698,  0.30385106,  0.21018787 };

T Mat_(rnd3x3T)[9] =
  { 0.07991805,  0.84907812,  0.66486698,
    0.97586422,  0.66531035,  0.30385106,
    0.2668651 ,  0.93157663,  0.21018787 };

T Mat_(pd3x3)[9] =
  { 0.29291524,  0.06772514,  0.00308962,
    0.06772514,  0.55844815,  0.02191619,
    0.00308962,  0.02191619,  0.02722591 };


AIATensor_ *mktnsr__(rnd4x4)(void) {
  return aiatensor__(newFromData)(mktnsr__(clone)(Mat_(rnd4x4), 16), shape4x4);
}

AIATensor_ *mktnsr__(rnd4x4T)(void) {
  return aiatensor__(newFromData)(mktnsr__(clone)(Mat_(rnd4x4T), 16), shape4x4);
}

AIATensor_ *mktnsr__(rnd3x3)(void) {
  return aiatensor__(newFromData)(mktnsr__(clone)(Mat_(rnd3x3), 9), shape3x3);
}

AIATensor_ *mktnsr__(rnd3x3T)(void) {
  return aiatensor__(newFromData)(mktnsr__(clone)(Mat_(rnd3x3T), 9), shape3x3);
}

AIATensor_ *mktnsr__(pd3x3)(void) {
  return aiatensor__(newFromData)(mktnsr__(clone)(Mat_(pd3x3), 9), shape3x3);
}


#endif

#define ERASE_FLOAT
#define ERASURE_FILE "sample_tensors.c"
#include <aianon/core/erasure.h>