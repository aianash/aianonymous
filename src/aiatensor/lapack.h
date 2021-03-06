#ifndef AIA_LAPACK_H

#include <aiautil/util.h>

// [TODO] using CMAKE
#ifndef USE_LAPACK
#define USE_LAPACK
#endif

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

#define aia_lapackCheckWithCleanup(fmt, cleanup, func, info, ...)     \
if (info < 0) {                                                       \
  cleanup                                                             \
  aia_error("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {                                                 \
  cleanup                                                             \
  aia_error(fmt, func, info, ##__VA_ARGS__);                          \
}

#endif

#ifdef ERASED_TYPE_PRESENT

/* AX=B */
AIA_API void aialapack__(gesv)(int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb, int* info);
/* Solve a triangular system of the form A * X = B  or A^T * X = B */
AIA_API void aialapack__(trtrs)(char uplo, char trans, char diag, int n, int nrhs, T *a, int lda, T *b, int ldb, int* info);
/* ||AX-B|| */
AIA_API void aialapack__(gels)(char trans, int m, int n, int nrhs, T *a, int lda, T *b, int ldb, T *work, int lwork, int *info);
/* Eigenvals */
AIA_API void aialapack__(syev)(char jobz, char uplo, int n, T *a, int lda, T *w, T *work, int lwork, int *info);
/* Non-sym eigenvals */
AIA_API void aialapack__(geev)(char jobvl, char jobvr, int n, T *a, int lda, T *wr, T *wi, T* vl, int ldvl, T *vr, int ldvr, T *work, int lwork, int *info);
/* svd */
AIA_API void aialapack__(gesvd)(char jobu, char jobvt, int m, int n, T *a, int lda, T *s, T *u, int ldu, T *vt, int ldvt, T *work, int lwork, int *info);
/* LU decomposition */
AIA_API void aialapack__(getrf)(int m, int n, T *a, int lda, int *ipiv, int *info);
/* Matrix Inverse */
AIA_API void aialapack__(getri)(int n, T *a, int lda, int *ipiv, T *work, int lwork, int* info);

/* Positive Definite matrices */
/* Cholesky factorization */
AIA_API void aialapack__(potrf)(char uplo, int n, T *a, int lda, int *info);
/* Matrix inverse based on Cholesky factorization */
AIA_API void aialapack__(potri)(char uplo, int n, T *a, int lda, int *info);
/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
AIA_API void aialapack__(potrs)(char uplo, int n, int nrhs, T *a, int lda, T *b, int ldb, int *info);
/* Cholesky factorization with complete pivoting. */
AIA_API void aialapack__(pstrf)(char uplo, int n, T *a, int lda, int *piv, int *rank, T tol, T *work, int *info);

/* QR decomposition */
AIA_API void aialapack__(geqrf)(int m, int n, T *a, int lda, T *tau, T *work, int lwork, int *info);
/* Build Q from output of geqrf */
AIA_API void aialapack__(orgqr)(int m, int n, int k, T *a, int lda, T *tau, T *work, int lwork, int *info);
/* Multiply Q with a matrix from output of geqrf */
AIA_API void aialapack__(ormqr)(char side, char trans, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int lwork, int *info);

#endif

#ifndef aialapack_
#define aialapack_(type, name) AIA_FN_ERASE_(lapack, type, name)
#define aialapack__(name) AIA_FN_ERASE_(lapack, T_, name)
#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiatensor/lapack.h"
#include <aiautil/erasure.h>


#define AIA_LAPACK_H
#endif