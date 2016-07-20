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
// sum of elements of a tensor along dimension
AIA_API void aiatensor__(sum)(AIATensor_ *res, AIATensor_ *tnsr, int dimension);
// gives sqare root of tensor values
AIA_API void aiatensor__(sqrt)(AIATensor_ *res, AIATensor_ *tnsr);
// gives exponential root of tensor values
AIA_API void aiatensor__(exp)(AIATensor_ *res, AIATensor_ *tnsr);
// gives log of tensor values
AIA_API void aiatensor__(log)(AIATensor_ *res, AIATensor_ *tnsr);
// gives ceiling value of tensor values
AIA_API void aiatensor__(ceil)(AIATensor_ *res, AIATensor_ *tnsr);
// gives floor value of tensor values
AIA_API void aiatensor__(floor)(AIATensor_ *res, AIATensor_ *tnsr);
// gives round value of tensor values
AIA_API void aiatensor__(round)(AIATensor_ *res, AIATensor_ *tnsr);
// gives absolute value of tensor values
AIA_API void aiatensor__(abs)(AIATensor_ *res, AIATensor_ *tnsr);
// gives truncated value of tensor values
AIA_API void aiatensor__(trunc)(AIATensor_ *res, AIATensor_ *tnsr);

AIA_API void aiatensor__(cadd)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2);
AIA_API void aiatensor__(csub)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2);
AIA_API void aiatensor__(cmul)(AIATensor_ *res, AIATensor_ *tnsr1, AIATensor_ *tnsr2);
AIA_API void aiatensor__(cpow)(AIATensor_ *res, AIATensor_ *base, AIATensor_ *exp);
AIA_API void aiatensor__(cdiv)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
AIA_API void aiatensor__(cfmod)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
AIA_API void aiatensor__(cremainder)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);

// elementwise multiplication of a repeated vector and matrix along column of matrix
AIA_API void aiatensor__(emulmv)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *vec);
// elementwise addition of a repeated vector and matrix along column of matrix
AIA_API void aiatensor__(eaddmv)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *vec);

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

#if defined(T_IS_FLOAT) || defined(T_IS_DOUBLE)
AIA_API int aiatensor__(epsieq)(AIATensor_ *a, AIATensor_ *b, T epsi);
#endif

AIA_API void aiatensor__(mm)(AIATensor_ *res, AIATensor_ *mat1, AIATensor_ *mat2);
AIA_API void aiatensor__(mv)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *vec);
AIA_API T aiatensor__(dot)(AIATensor_ *vec1, AIATensor_ *vec2);

AIA_API T aiatensor__(trace)(AIATensor_ *mat);

AIA_API void aiatensor__(fill)(AIATensor_ *res, T value);
AIA_API void aiatensor__(zero)(AIATensor_ *res);
AIA_API void aiatensor__(maskedFill)(AIATensor_ *res, AIATensor(uchar) *mask, T value);

AIA_API void aiatensor__(zeros)(AIATensor_ *res, int nDimension, long *size, long *stride);
AIA_API void aiatensor__(ones)(AIATensor_ *res, int nDimension, long *size, long *stride);

AIA_API void aiatensor__(sqrt)(AIATensor_ * res, AIATensor_ *tnsr);

#if defined(T_IS_DOUBLE) || defined(T_IS_FLOAT)
/**
 * Description
 * -----------
 * Computes determinant of a positive definite matrix
 *
 * Input
 * -----
 * mat : Positive definite matrix of size n x n
 *
 * Output
 * ------
 * Returns determinant of matrix
 */
AIA_API T aiatensor__(detpd)(AIATensor_ *mat);
#endif

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
 *   x.T * amat * y
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
 *   x.T * amat * x
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
 *   x.T * A^-1 * x
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
 *   x.T * A^-1 * y
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
 *   x.T * A * x
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
 *   x.T * A * y
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

#if defined(T_IS_DOUBLE) || defined(T_IS_FLOAT)
/**
 * Description
 * -----------
 * Returns following product for a positive definite matrix A
 *   x.T * A^-1 * x
 *
 * Input
 * -----
 * x     : Vector of size n
 * achol : Cholesky factorization of a positive definite matrix of size n x n
 * uplo  : "U" or "L" depending on whether achol has upper or lower triangular matrix
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTApdIx)(AIATensor_ *x, AIATensor_ *achol, const char *uplo);
#endif

#if defined(T_IS_DOUBLE) || defined(T_IS_FLOAT)
/**
 * Description
 * -----------
 * Returns following product for a positive definite matrix A
 *   x.T * A^-1 * y
 *
 * Input
 * -----
 * x     : Vector of size n
 * achol : Cholesky factorization of a positive definite matrix of size n x n
 * y     : Vector of size n
 * uplo  : "U" or "L" depending on whether achol has upper or lower triangular matrix
 *
 * Output
 * ------
 * Returns a scalar
 */
AIA_API T aiatensor__(xTApdIy)(AIATensor_ *x, AIATensor_ *achol, const char *uplo, AIATensor_ *y);
#endif

/**
 * Description
 * -----------
 * Computes following matrix-matrix multiplication
 *   X.T * A * X + a * Y
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

#if defined(T_IS_DOUBLE) || defined(T_IS_FLOAT)
/**
 * Description
 * -----------
 * Computes following matrix-matrix multiplication for positive definite matrix A
 *   X.T * A^-1 * X + a * Y
 *
 * Input
 * -----
 * xmat  : Matrix of size n x n
 * achol : Cholesky factorization of positive definite matrix of size n x n
 * uplo  : "U" or "L" depending on whether achol has upper or lower triangular matrix
 * a     : Multiplication factor
 * ymat  : Matrix of size n x n
 *
 * Output
 * ------
 * res  : Matrix of size n x n
 * If res is NULL, it creates a new matrix. Client has to delete the returned matrix.
 *
 */
AIA_API AIATensor_ *aiatensor__(XTApdIXpaY)(AIATensor_ *res, AIATensor_ *xmat, AIATensor_ *achol, const char *uplo, T a, AIATensor_ *ymat);
#endif

#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor/math.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_MATH_H
#endif