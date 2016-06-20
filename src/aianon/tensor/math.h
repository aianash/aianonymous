#ifndef AIA_TENSOR_MATH_H

#include <aianon/core/util.h>
#include <aianon/tensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

// res = (beta * bvec) + (alpha * (mat * vec))
void aiatensor__(addmv)(AIATensor_ *res, T beta, AIATensor_ *bvec, T alpha, AIATensor_ *mat, AIATensor_ *vec);

// res = (beta * bmat) + (alpha * mat1 * mat2)
void aiatensor__(addmm)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *mat1, AIATensor_ *mat2);

// res = (beta * bmat) + (alpha * vec1 x vec2)
void aiatensor__(addr)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *vec1, AIATensor_ *vec2);



// [TODO] REMOVE
void aiablas__(gemv)(char trans, long m, long n, T alpha, T *a, long lda, T *x, long incx, T beta, T *y, long incy) {}
void aiablas__(ger)(long m, long n, T alpha, T *x, long incx, T *y, long incy, T *a, long lda) {}
/* Level 3 */
void aiablas__(gemm)(char transa, char transb, long m, long n, long k, T alpha, T *a, long lda, T *b, long ldb, T beta, T *c, long ldc) {}

void aiatensor__(mul)(AIATensor_ *r_, AIATensor_ *t, T value) {}

#endif

// [TODO] REMOVE
#ifndef aiablas__
#define aiablas__(name) AIA_FN_ERASE_(blas, T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/math.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_MATH_H
#endif