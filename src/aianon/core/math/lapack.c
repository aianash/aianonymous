#include <aianon/core/math/lapack.h>

#ifdef ERASED_TYPE_PRESENT

#undef DS_
#ifdef T_IS_DOUBLE
# define DS_ d
#else
# define DS_ s
#endif

#define ds_(name) AIA_CONCAT_2(DS_, name)

extern void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern void strtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);
extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
extern void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
extern void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *info);
extern void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
extern void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
extern void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
extern void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
extern void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);
extern void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
extern void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
extern void spotri_(char *uplo, int *n, float *a, int *lda, int *info);
extern void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);
extern void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
extern void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
extern void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);
extern void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern void spstrf_(char *uplo, int *n, float *a, int *lda, int *piv, int *rank, float *tol, float *work, int *info);
extern void dpstrf_(char *uplo, int *n, double *a, int *lda, int *piv, int *rank, double *tol, double *work, int *info);

/* Compute the solution to a real system of linear equations  A * X = B */
void aialapack__(gesv)(int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb, int* info) {
#ifdef USE_LAPACK
  ds_(gesv_)(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve a triangular system of the form A * X = B  or A^T * X = B */
void aialapack__(trtrs)(char uplo, char trans, char diag, int n, int nrhs, T *a, int lda, T *b, int ldb, int* info) {
#ifdef USE_LAPACK
  ds_(trtrs_)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  THError("trtrs : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve overdetermined or underdetermined real linear systems involving an
M-by-N matrix A, or its transpose, using a QR or LQ factorization of A */
void aialapack__(gels)(char trans, int m, int n, int nrhs, T *a, int lda, T *b, int ldb, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(gels_)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

/* Compute all eigenvalues and, optionally, eigenvectors of a real symmetric
matrix A */
void aialapack__(syev)(char jobz, char uplo, int n, T *a, int lda, T *w, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(syev_)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  THError("syev : Lapack library not found in compile time\n");
#endif
}

/* Compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and,
optionally, the left and/or right eigenvectors */
void aialapack__(geev)(char jobvl, char jobvr, int n, T *a, int lda, T *wr, T *wi, T *vl, int ldvl, T *vr, int ldvr, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(geev_)(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#else
  THError("geev : Lapack library not found in compile time\n");
#endif
}

/* Compute the singular value decomposition (SVD) of a real M-by-N matrix A,
optionally computing the left and/or right singular vectors */
void aialapack__(gesvd)(char jobu, char jobvt, int m, int n, T *a, int lda, T *s, T *u, int ldu, T *vt, int ldvt, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(gesvd_)( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void aialapack__(getrf)(int m, int n, T *a, int lda, int *ipiv, int *info) {
#ifdef USE_LAPACK
  ds_(getrf_)(&m, &n, a, &lda, ipiv, info);
#else
  THError("getrf : Lapack library not found in compile time\n");
#endif
}
/* Matrix Inverse */
void aialapack__(getri)(int n, T *a, int lda, int *ipiv, T *work, int lwork, int* info) {
#ifdef USE_LAPACK
  ds_(getri_)(&n, a, &lda, ipiv, work, &lwork, info);
#else
  THError("getri : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization */
void aialapack__(potrf)(char uplo, int n, T *a, int lda, int *info) {
#ifdef USE_LAPACK
  ds_(potrf_)(&uplo, &n, a, &lda, info);
#else
  THError("potrf : Lapack library not found in compile time\n");
#endif
}

/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void aialapack__(potrs)(char uplo, int n, int nrhs, T *a, int lda, T *b, int ldb, int *info) {
#ifdef USE_LAPACK
  ds_(potrs_)(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  THError("potrs: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization based Matrix Inverse */
void aialapack__(potri)(char uplo, int n, T *a, int lda, int *info) {
#ifdef USE_LAPACK
  ds_(potri_)(&uplo, &n, a, &lda, info);
#else
  THError("potri: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization with complete pivoting */
void aialapack__(pstrf)(char uplo, int n, T *a, int lda, int *piv, int *rank, T tol, T *work, int *info) {
#ifdef USE_LAPACK
  ds_(pstrf_)(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#else
  THError("pstrf: Lapack library not found at compile time\n");
#endif
}

/* QR decomposition */
void aialapack__(geqrf)(int m, int n, T *a, int lda, T *tau, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(geqrf_)(&m, &n, a, &lda, tau, work, &lwork, info);
#else
  THError("geqrf: Lapack library not found in compile time\n");
#endif
}

/* Build Q from output of geqrf */
void aialapack__(orgqr)(int m, int n, int k, T *a, int lda, T *tau, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(orgqr_)(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#else
  THError("orgqr: Lapack library not found in compile time\n");
#endif
}

/* Multiply Q with a matrix using the output of geqrf */
void aialapack__(ormqr)(char side, char trans, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int lwork, int *info) {
#ifdef USE_LAPACK
  ds_(ormqr_)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#else
  THError("ormqr: Lapack library not found in compile time\n");
#endif
}


#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/core/math/lapack.c"
#include <aiautil/erasure.h>