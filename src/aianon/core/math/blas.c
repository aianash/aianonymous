#include <aianon/core/math/blas.h>

#ifdef ERASED_TYPE_PRESENT

#undef DS_
#ifdef T_IS_DOUBLE
# define DS_ d
#else
# define DS_ s
#endif

#define ds_(name) AIA_CONCAT_2(DS_, name)

extern void dswap_(int *n, double *x, int *incx, double *y, int *incy);
extern void sswap_(int *n, float *x, int *incx, float *y, int *incy);
extern void dscal_(int *n, double *a, double *x, int *incx);
extern void sscal_(int *n, float *a, float *x, int *incx);
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void scopy_(int *n, float *x, int *incx, float *y, int *incy);
extern void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);
extern void saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern float sdot_(int *n, float *x, int *incx, float *y, int *incy);
extern void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
extern void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);


void aiablas__(swap)(long n, T *x, long incx, T *y, long incy) {
  if(n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int) n;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    ds_(swap_)(&i_n, x, &i_incx, y, &i_incy);
  }
#else
#  warning swap: Blas library not found in compile time.
#endif
}


void aiablas__(scal)(long n, T a, T *x, long incx) {
  if(n == 1) incx = 1;

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((n <= INT_MAX) && (incx <= INT_MAX)) {
    int i_n = (int) n;
    int i_incx = (int) incx;
    ds_(scal_)(&i_n, &a, x, &i_incx);
  }
#else
# warning scal: Blas library not found in compile time.
#endif
}


void aiablas__(copy)(long n, T *x, long incx, T *y, long incy) {
  if(n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int) n;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    ds_(copy_)(&i_n, x, &i_incx, y, &i_incy);
  }
#else
#  warning copy: Blas library not found in compile time.
#endif
}


void aiablas__(axpy)(long n, T a, T *x, long incx, T *y, long incy) {
  if(n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int) n;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    ds_(axpy_)(&i_n, &a, x, &i_incx, y, &i_incy);
  }
#else
#  warning axpy: Blas library not found in compile time.
#endif
}


T aiablas__(dot)(long n, T *x, long incx, T *y, long incy) {
  if(n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int) n;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    return (T) ds_(dot_)(&i_n, x, &i_incx, y, &i_incy);
  }
#else
#  warning dot: Blas library not found in compile time.
#endif
}


void aiablas__(gemv)(char trans, long m, long n, T alpha, T *a, long lda, T *x, long incx, T beta, T *y, long incy) {
  if(n == 1) lda = m;

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((m <= INT_MAX) && (n <= INT_MAX) && (lda > 0) && (lda <= INT_MAX)
    && (incx > 0) && (incx <= INT_MAX) && (incy > 0) && (incy <= INT_MAX)) {
    int i_m = (int) m;
    int i_n = (int) n;
    int i_lda = (int) lda;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    ds_(gemv_)(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
  }
#else
#  warning gemv: Blas library not found in compile time.
#endif
}


void aiablas__(ger)(long m, long n, T alpha, T *x, long incx, T *y, long incy, T *a, long lda) {
  if(n == 1) lda = m;

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)
    && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_m = (int) m;
    int i_n = (int) n;
    int i_lda = (int) lda;
    int i_incx = (int) incx;
    int i_incy = (int) incy;
    ds_(ger_)(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
  }
#else
#  warning ger: Blas library not found in compile time.
#endif
}


void aiablas__(gemm)(char transa, char transb, long m, long n, long k, T alpha, T *a, long lda, T *b, long ldb, T beta, T *c, long ldc) {
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1) ldc = m;

  if(transa_) {
    if(m == 1) lda = k;
  } else {
    if(k == 1) lda = m;
  }

  if(transb_) {
    if(k == 1) ldb = n;
  } else {
    if(n == 1) ldb = k;
  }

#if defined(USE_BLAS) && (defined(T_IS_DOUBLE) || defined(T_IS_FLOAT))
  if((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX)
    && (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX)) {
    int i_m = (int) m;
    int i_n = (int) n;
    int i_k = (int) k;
    int i_lda = (int) lda;
    int i_ldb = (int) ldb;
    int i_ldc = (int) ldc;
    ds_(gemm_)(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
  }
#else
#  warning gemm: Blas library not found in compile time
#endif
}

#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/core/math/blas.c"
#include <aianon/core/erasure.h>