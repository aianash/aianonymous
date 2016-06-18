#include <limits.h>

#ifndef AIA_BLAS_H
#ifdef ERASED_TYPE_AVAILABLE

// [TODO] using CMAKE
#ifndef USE_BLAS
#define USE_BLAS
#endif

// [TODO] write proper description with parameter details
/** Swap two tensors */
extern void aiablas_(T_, swap)(long n, T *x, long incx, T *y, long incy);

/** Scale a tensor x <- ax */
extern void aiablas_(T_, scal)(long n, T a, T *x, long incx);

/** Copy x into y */
extern void aiablas_(T_, copy)(long n, T *x, long incx, T *y, long incy);

/** x <- ax + y */
extern void aiablas_(T_, axpy)(long n, T a, T *x, long incx, T *y, long incy);

/** Dot product of two tensors */
extern T aiablas_(T_, dot)(long n, T *x, long incx, T *y, long incy);

/** Matrix vector multiplication */
extern void aiablas_(T_, gemv)(char trans, long m, long n, T alpha, T *a, long lda, T *x, long incx, T beta, T *y, long incy);

/** Performs rank 1 operation A := alpha*x*y**T + A */
extern void aiablas_(T_, ger)(long m, long n, T alpha, T *x, long incx, T *y, long incy, T *a, long lda);

/** Matrix matrix multiplication */
extern void aiablas_(T_, gemm)(char transa, char transb, long m, long n, long k, T alpha, T *a, long lda, T *b, long ldb, T beta, T *c, long ldc);

#endif

#ifndef aiablas_
#define aiablas_(type, name) AIA_FN_ERASE_(blas, type, name)
#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/math/blas.h"
#include <aianon/util/erasure.h>


#define AIA_BLAS_H
#endif