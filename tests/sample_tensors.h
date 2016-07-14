#ifndef SAMPLE_TENSORS_H

#include <aianon/tensor/tensor.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

extern TensorShape shape4x4;
extern TensorShape shape3x3;
#endif

#ifdef ERASED_TYPE_PRESENT

extern T Mat_(rnd4x4)[16];
extern T Mat_(rnd4x4T)[16];
extern T Mat_(rnd3x3)[9];
extern T Mat_(rnd3x3T)[9];
extern T Mat_(pd3x3)[9];

AIATensor_ *mktnsr__(rnd4x4)(void);
AIATensor_ *mktnsr__(rnd4x4T)(void);

AIATensor_ *mktnsr__(rnd3x3)(void);
AIATensor_ *mktnsr__(rnd3x3T)(void);

AIATensor_ *mktnsr__(pd3x3)(void);

#endif

#ifndef Mat
#define mktnsr_(type, name) AIA_FN_ERASE_(mktnsr, type, name)
#define Mat(type, name) AIA_CONCAT_3(T_, Mat, name)
#define Vec(type, name) AIA_CONCAT_3(T_, Vec, name)

#define mktnsr__(name) mktnsr_(T_, name)
#define Mat_(name) Mat(T_, name)
#define Vec_(name) Vec(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "sample_tensors.h"
#include <aianon/core/erasure.h>

#define SAMPLE_TENSORS_H
#endif