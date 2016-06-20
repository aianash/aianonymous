#ifndef AIA_BLAS_H

#include <limits.h>
#include <aianon/core/util.h>

#ifdef ERASED_TYPE_PRESENT

// [TODO] using CMAKE
#ifndef USE_BLAS
#define USE_BLAS
#endif

/**
 * Function: aiablas_(T_, swap)
 * ---------------------------
 * Swaps two vectors x and y
 *
 * n    : number of elements in input vector(s)
 * x    : vector with n elements
 * incx : storage spaces between elements of x
 * y    : vector with n elements
 * incy : storage spaces between elements of y
 */
AIA_API void aiablas_(T_, swap)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas_(T_, scal)
 * ----------------------------
 * Scales a vector by a constant
 *
 * n    : number of elements in given vector
 * a    : scaling constant
 * x    : vector with n elements
 * incx : storage spaces between elements of y
 */
AIA_API void aiablas_(T_, scal)(long n, T a, T *x, long incx);

/**
 * Function: aiablas_(T_, copy)
 * ----------------------------
 * Copies vector x to vector y
 *
 * n    : number of elements in given vector
 * x    : vector with n elements
 * incx : storage spaces between elements of x
 * y    : vector with n elements
 * incy : storage spaces between elements of y
 */
AIA_API void aiablas_(T_, copy)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas_(T_, axpy)
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
AIA_API void aiablas_(T_, axpy)(long n, T a, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas_(T_, dot)
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
AIA_API T aiablas_(T_, dot)(long n, T *x, long incx, T *y, long incy);

/**
 * Function: aiablas_(T_, gemv)
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
AIA_API void aiablas_(T_, gemv)(char trans, long m, long n, T alpha, T *a, long lda, T *x, long incx, T beta, T *y, long incy);

/**
 * Function: aiablas_(T_, ger)
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
AIA_API void aiablas_(T_, ger)(long m, long n, T alpha, T *x, long incx, T *y, long incy, T *a, long lda);

/**
 * Function: aiablas_(T_, gemm)
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
AIA_API void aiablas_(T_, gemm)(char transa, char transb, long m, long n, long k, T alpha, T *a, long lda, T *b, long ldb, T beta, T *c, long ldc);

#endif

#ifndef aiablas_
#define aiablas_(type, name) AIA_FN_ERASE_(blas, type, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/core/math/blas.h"
#include <aianon/core/erasure.h>


#define AIA_BLAS_H
#endif