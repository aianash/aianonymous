#ifndef AIA_LAPACK_H
#ifdef ERASED_TYPE_AVAILABLE

// [TODO] using CMAKE
#ifndef USE_LAPACK
#define USE_LAPACK
#endif

/* AX=B */
void aialapack_(T_, gesv)(int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb, int* info);
/* Solve a triangular system of the form A * X = B  or A^T * X = B */
void aialapack_(T_, trtrs)(char uplo, char trans, char diag, int n, int nrhs, T *a, int lda, T *b, int ldb, int* info);
/* ||AX-B|| */
void aialapack_(T_, gels)(char trans, int m, int n, int nrhs, T *a, int lda, T *b, int ldb, T *work, int lwork, int *info);
/* Eigenvals */
void aialapack_(T_, syev)(char jobz, char uplo, int n, T *a, int lda, T *w, T *work, int lwork, int *info);
/* Non-sym eigenvals */
void aialapack_(T_, geev)(char jobvl, char jobvr, int n, T *a, int lda, T *wr, T *wi, T* vl, int ldvl, T *vr, int ldvr, T *work, int lwork, int *info);
/* svd */
void aialapack_(T_, gesvd)(char jobu, char jobvt, int m, int n, T *a, int lda, T *s, T *u, int ldu, T *vt, int ldvt, T *work, int lwork, int *info);
/* LU decomposition */
void aialapack_(T_, getrf)(int m, int n, T *a, int lda, int *ipiv, int *info);
/* Matrix Inverse */
void aialapack_(T_, getri)(int n, T *a, int lda, int *ipiv, T *work, int lwork, int* info);

/* Positive Definite matrices */
/* Cholesky factorization */
void aialapack_(T_, potrf)(char uplo, int n, T *a, int lda, int *info);
/* Matrix inverse based on Cholesky factorization */
void aialapack_(T_, potri)(char uplo, int n, T *a, int lda, int *info);
/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void aialapack_(T_, potrs)(char uplo, int n, int nrhs, T *a, int lda, T *b, int ldb, int *info);
/* Cholesky factorization with complete pivoting. */
void aialapack_(T_, pstrf)(char uplo, int n, T *a, int lda, int *piv, int *rank, T tol, T *work, int *info);

/* QR decomposition */
void aialapack_(T_, geqrf)(int m, int n, T *a, int lda, T *tau, T *work, int lwork, int *info);
/* Build Q from output of geqrf */
void aialapack_(T_, orgqr)(int m, int n, int k, T *a, int lda, T *tau, T *work, int lwork, int *info);
/* Multiply Q with a matrix from output of geqrf */
void aialapack_(T_, ormqr)(char side, char trans, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int lwork, int *info);

#endif

#ifndef aialapack_
#define aialapack_(type, name) AIA_FN_ERASE_(lapack, type, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/math/lapack.h"
#include <aianon/util/erasure.h>


#define AIA_LAPACK_H
#endif