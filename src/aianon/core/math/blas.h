#ifndef AIA_BLAS_H

#include <limits.h>
#include <aianon/core/util.h>

#ifdef ERASED_TYPE_PRESENT

// [TODO] using CMAKE
#ifndef USE_BLAS
#define USE_BLAS
#endif

/**
 * Function: aiablas__(swap)
 * ---------------------------
 * Swaps two vectors x and y
 *
 * n    : number of elements in input vector(s)
 * x    : vector with n elements
 * incx : storage spaces between elements of x
 * y    : vector with n elements
 * incy : storage spaces between elements of y
 */
AIA_API void aiablas__(swap)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas__(scal)
 * ----------------------------
 * Scales a vector by a constant
 *
 * n    : number of elements in given vector
 * a    : scaling constant
 * x    : vector with n elements
 * incx : storage spaces between elements of y
 */
AIA_API void aiablas__(scal)(long n, T a, T *x, long incx);

/**
 * Function: aiablas__(copy)
 * ----------------------------
 * Copies vector x to vector y
 *
 * n    : number of elements in given vector
 * x    : vector with n elements
 * incx : storage spaces between elements of x
 * y    : vector with n elements
 * incy : storage spaces between elements of y
 */
AIA_API void aiablas__(copy)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas__(axpy)
 * ----------------------------
 * Constant times a vector plus vector
 *   y := a * x + y
 *
 * n    : number of elements in given vector
 * a    : scaling constant
 * x    : vector with n elements
 * incx : storage spaces between elements of x
 * y    : vector with n elements
 * incy : storage spaces between elements of y
 */
AIA_API void aiablas__(axpy)(long n, T a, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas__(dot)
 * ---------------------------
 * Dot product of two vectors
 *
 * n       : number of elements in given vector
 * x       : vector with n elements
 * incx    : storage spaces between elements of x
 * y       : vector with n elements
 * incy    : storage spaces between elements of y
 *
 * returns : vector of n elements
 */
AIA_API T aiablas__(dot)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas__(gemv)
 * ----------------------------
 * Performs one of the matrix-vector operations
 *   y := alpha * A * x + beta * y   OR   y := alpha * A ** T * x + beta * y
 *
 * trans :
 * m     : number of rows in matrix a
 * n     : number of columns in matrix a
 * alpha : scalar alpha
 * a     :
 * lda   :
 * x     :
 * incx  :
 * beta  :
 * y     :
 * incy  :
 */
AIA_API void aiablas__(gemv)(char trans, long m, long n, T alpha, T *a, long lda, T *x, long incx, T beta, T *y, long incy);

/**
 * Function: aiablas__(ger)
 * ---------------------------
 * Performs rank 1 operation
 *   a := alpha * x * y**T + A
 *
 * m     :
 * n     :
 * alpha :
 * x     :
 * incx  :
 * y     :
 * incy  :
 * a     :
 * lda   :
 */
AIA_API void aiablas__(ger)(long m, long n, T alpha, T *x, long incx, T *y, long incy, T *a, long lda);

/**
 * Function: aiablas__(gemm)
 * ----------------------------
 * Performs one of the matrix-matrix operation
 *   C := alpha * op(A) * op(B) + beta * C
 * where  op( X ) is one of
 *   op(X) = X   or   op(X) = X**T
 *
 * transa :
 * transb :
 * m      :
 * n      :
 * k      :
 * alpha  :
 * a      :
 * lda    :
 * b      :
 * ldb    :
 * beta   :
 * c      :
 * ldc    :
 */
AIA_API void aiablas__(gemm)(char transa, char transb, long m, long n, long k, T alpha, T *a, long lda, T *b, long ldb, T beta, T *c, long ldc);

#endif

#ifndef aiablas_
#define aiablas_(type, name) AIA_FN_ERASE_(blas, type, name)
#define aiablas__(name) aiablas_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/core/math/blas.h"
#include <aianon/core/erasure.h>


#define AIA_BLAS_H
#endif