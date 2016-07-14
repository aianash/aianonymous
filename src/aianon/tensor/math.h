#ifndef AIA_TENSOR_MATH_H

#include <aianon/core/util.h>
#include <aianon/tensor/tensor.h>
#include <aianon/core/math/blas.h>
#include <aianon/tensor/dimapply.h>
#include <aianon/tensor/apply.h>

#define AIA_OMP_OVERHEAD_THRESHOLD 100000

#ifdef ERASED_TYPE_PRESENT

// res = tnsr(element) + value
AIA_API void aiatensor__(add)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// res = tnsr(element) - value
AIA_API void aiatensor__(sub)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// res = tnsr(element) * value
AIA_API void aiatensor__(mul)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// res = tnsr(element) / value
AIA_API void aiatensor__(div)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// res = tnsr(element) % value (uses c fmod function to calculate remainder)
AIA_API void aiatensor__(fmod)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// res = tnsr(element) % value (uses (a % b = a - b * floor(a/b)) to calculate remainder)
AIA_API void aiatensor__(remainder)(AIATensor_ *res, AIATensor_ *tnsr, T value);
// Clamp all elements in the Tensor into the range [min_value, max_value]
AIA_API void aiatensor__(clamp)(AIATensor_ *res, AIATensor_ *tnsr, T minValue, T maxValue);

AIA_API void aiatensor__(cadd)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2);
AIA_API void aiatensor__(csub)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2);
AIA_API void aiatensor__(cmul)(AIATensor_ *res, AIATensor_ *tnsr1, AIATensor_ *tnsr2);
AIA_API void aiatensor__(cpow)(AIATensor_ *res, AIATensor_ *base, AIATensor_ *exp);
AIA_API void aiatensor__(cdiv)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
AIA_API void aiatensor__(cfmod)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
AIA_API void aiatensor__(cremainder)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);

// res = tnsr1 + alpha * (tnsr2 * tnsr3)
AIA_API void aiatensor__(addcmul)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2, AIATensor_ *tnsr3);
// res = tnsr1 + alpha * (tnsr2 / tnsr3)
AIA_API void aiatensor__(addcdiv)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tndr2, AIATensor_ *tnsr3);

// res = (beta * bvec) + (alpha * (mat * vec))
void aiatensor__(addmv)(AIATensor_ *res, T beta, AIATensor_ *bvec, T alpha, AIATensor_ *mat, AIATensor_ *vec);

// res = (beta * bmat) + (alpha * mat1 * mat2)
void aiatensor__(addmm)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *mat1, AIATensor_ *mat2);

// res = (beta * bmat) + (alpha * vec1 x vec2)
void aiatensor__(addr)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *vec1, AIATensor_ *vec2);

AIA_API void aiatensor__(addbmm)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *batch1, AIATensor_ *batch2);
AIA_API void aiatensor__(baddbmm)(AIATensor_ *res, T beta, AIATensor_ *batch3, T alpha, AIATensor_ *batch1, AIATensor_ *batch2);

AIA_API int aiatensor__(eq)(AIATensor_ *a, AIATensor_ *b);
AIA_API int aiatensor__(epsieq)(AIATensor_ *a, AIATensor_ *b, T epsi);

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/math.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_MATH_H
#endif