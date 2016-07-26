#ifndef AIA_TENSOR_LINALG_H

#include <aiautil/util.h>
#include <aiatensor/lapack.h>
#include <aiatensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

/**
 * Description
 * -----------
 * Computes cholesky factorization of a positive definite matrix
 *
 * Input
 * -----
 * mat  : Positive definite matrix of size n x n
 * uplo : "U" or "L"
 *
 * Output
 * ------
 * res  : Cholesky factorization
 */
AIA_API void aiatensor__(potrf)(AIATensor_ *res, AIATensor_ *mat, const char *uplo);

/**
 * Description
 * -----------
 * Computes solution of linear system of equations Ax = B where A
 * is positive definite
 *
 * Input
 * -----
 * b    : Matrix of size n x p or vector of size n
 * a    : Cholesky factorization of ositive definite matrix of size n x n
 *        as obtained from portf
 * uplo : "U" or "L"
 *
 * Output
 * ------
 * res  : Matrix of size n x p or vector of size n
 */
AIA_API void aiatensor__(potrs)(AIATensor_ *res, AIATensor_ *b, AIATensor_ *a, const char *uplo);

/**
 * Description
 * -----------
 * Solves triangular system of form
 *   A * x = B   or   A.T * x = B
 * Input
 * -----
 * b     : Matrix of size n x p or vector of size n
 * amat  : Triangular matrix of size n x n
 * uplo  : "U" or "L"
 * trans : "N" for A * x = B
 *         "T" for A.T * x = B
 * diag  : "N" if A is non-unit triangular
 *         "U" if A is unit triangular
 *
 * Output
 * ------
 * resa  : Matrix of size n x n for temporary storage
 * resb  : Matrix of size n x p or vector of size n
 */
AIA_API void aiatensor__(trtrs)(AIATensor_ *resa, AIATensor_ *resb, AIATensor_ *b, AIATensor_ *amat, const char *uplo, const char *trans, const char *diag);

/**
 * Description
 * -----------
 * Computes the SVD of a matrix of size n x m
 *   A = U * sigma * V.T
 *
 * Input
 * -----
 * mat  : Matrix of size n x m
 * jobu : "A" if all singular values are to be computed
 *        "S" if some singular values are to be computed
 *
 * Output
 * ------
 * resu : Matrix of size n x n
 * ress : Matrix of size n x m
 * resv : Matrix of size m x m
 */
AIA_API void aiatensor__(gesvd)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *mat, const char *jobu);

/**
 * Description
 * -----------
 * Computes the SVD of a matrix of size n x m
 *   A = U * sigma * V.T
 *
 * Input
 * -----
 * mat  : Matrix of size n x m
 * jobu : "A" if all singular values are to be computed
 *        "S" if some singular values are to be computed
 *
 * Output
 * ------
 * rmat :
 * resu : Matrix of size n x n
 * ress : Matrix of size n x m
 * resv : Matrix of size m x m
 */
AIA_API void aiatensor__(gesvd2)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *rmat, AIATensor_ *mat, const char *jobu);

/**
 * Description
 * -----------
 * Computes all eigenvalues and eigenvectors of a symmetric matrix A
 *
 * Input
 * -----
 * mat  : Matrix of size n x n
 * jobz : "N" if only eignevalues are to be computed
 *        "L" if both eignevalues and eigenvectors are to be computed
 * uplo : "U" if upper triangular part of A is stored
 *        "L" if lower triangular part of A is stored
 *
 * Output
 * ------
 * rese : Vector of size n containing eigenvalies
 * resv : Matrix of size n x n with columns as eigenvectors
 */
AIA_API void aiatensor__(syev)(AIATensor_ *rese, AIATensor_ *resv, AIATensor_ *mat, const char *jobz, const char *uplo);

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiatensor/linalg.h"
#include <aiautil/erasure.h>

#define AIA_TENSOR_LINALG_H
#endif