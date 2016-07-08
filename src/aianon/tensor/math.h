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

AIA_API void aiatensor__(mul)(AIATensor_ *r_, AIATensor_ *t, T value);

AIA_API void aiatensor__(mm)(AIATensor_ *res, AIATensor_ *mat1, AIATensor_ *mat2);
AIA_API void aiatensor__(mv)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *vec);
AIA_API T aiatensor__(dot)(AIATensor_ *vec1, AIATensor_ *vec2);

AIA_API T aiatensor__(trace)(AIATensor_ *mat);

/**
 * Description
 * -----------
 * Computes determinant of a symmetric matrix
 *
 * Input
 * -----
 * mat : Symmetric matrix of size n x n
 *
 * Output
 * ------
 * Returns determinant of matrix
 */
AIA_API T aiatensor__(detsymm)(AIATensor_ *mat);

/**
 * Description
 * -----------
 * Computes the following:
 *   res = mat + a * I
 *   Here I is identity matrix
 *
 * Input
 * -----
 * mat : Matrix of size n x n
 * a   : Multiplication factor
 *
 * Output
 * ------
 * res : Matrix of size n x n
 */
AIA_API void aiatensor__(aIpX)(AIATensor_ *res, AIATensor_ *mat, T a);

/**
 * Description
 * -----------
 * Returns following product:
 *   x** * amat * y
 *
 * Input
 * -----
 * x    : Vector of size m
 * amat : Matrix of size m x n
 * y    : Vector of size n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y);

/**
 * Description
 * -----------
 * Returns following product:
 *   x** * amat * x
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : Matrix of size n x n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAx)(AIATensor_ *x, AIATensor_ *amat);

/**
 * Description
 * -----------
 * Returns following product
 *   x** * A^-1 * x
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : LU factorization of matrix A where A is of size n x n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAIx)(AIATensor_ *x, AIATensor_ *amat);

/**
 * Description
 * -----------
 * Returns following product:
 *   x** * A^-1 * y
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : LU factorization of matrix A where A is of size n x n
 * y    : Vector of size n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAIy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y);

/**
 * Description
 * -----------
 * Returns following product for a symmetric matrix A
 *   x** * A * x
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : Symmetric matrix of size n x n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAsymmx)(AIATensor_ *x, AIATensor_ *amat);

/**
 * Description
 * -----------
 * Returns following product for a symmetric matrix A
 *   x** * A * y
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : Symmetric matrix of size n x n
 * y    : Vector of size n
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAsymmy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y);

/**
 * Description
 * -----------
 * Returns following product for a symmetric matrix A
 *   x** * A^-1 * x
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : Cholesky factorization of a symmetric matrix of size n x n
 * uplo : "U" or "L" depending on whether amat has upper or lower triangular matrix
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAsymmIx)(AIATensor_ *x, AIATensor_ *amat, const char *uplo);

/**
 * Description
 * -----------
 * Returns following product for a symmetric matrix A
 *   x** * A^-1 * y
 *
 * Input
 * -----
 * x    : Vector of size n
 * amat : Cholesky factorization of a symmetric matrix of size n x n
 * y    : Vector of size n
 * uplo : "U" or "L" depending on whether amat has upper or lower triangular matrix
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTAsymmIy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y, const char *uplo);

/**
 * Description
 * -----------
 * Computes following matrix-matrix multiplication
 *   X** * A * X + a * Y
 *
 * Input
 * -----
 * xmat : Matrix of size n x n
 * amat : Symmetric matrix of size n x n
 * a    : Multiplication factor
 * ymat : Matrix of size n x n
 *
 * Output
 * ------
 * res  : Matrix of size n x n
 * If res is NULL, it creates a new matrix. Client has to delete the returned matrix.
 *
 */
AIA_API AIATensor_ *aiatensor__(XTAsymmXpaY)(AIATensor_ *res, AIATensor_ *xmat, AIATensor_ *amat, T a, AIATensor_ *ymat);

/**
 * Description
 * -----------
 * Computes following matrix-matrix multiplication
 *   X** * A^-1 * X + a * Y
 *
 * Input
 * -----
 * xmat : Matrix of size n x n
 * amat : Cholesky factorization of symmetric matrix of size n x n
 * uplo : "U" or "L" depending on whether amat has upper or lower triangular matrix
 * a    : Multiplication factor
 * ymat : Matrix of size n x n
 *
 * Output
 * ------
 * res  : Matrix of size n x n
 * If res is NULL, it creates a new matrix. Client has to delete the returned matrix.
 *
 */
AIA_API AIATensor_ *aiatensor__(XTAsymmIXpaY)(AIATensor_ *res, AIATensor_ *xmat, AIATensor_ *amat, const char *uplo, T a, AIATensor_ *ymat);

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/math.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_MATH_H
#endif