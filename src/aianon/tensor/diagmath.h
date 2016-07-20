#ifndef AIA_TENSOR_DIAGMATH_H

#include <aianon/core/util.h>
#include <aianon/core/math/blas.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/dimapply.h>

#ifdef ERASED_TYPE_PRESENT

/**
 * Description
 * -----------
 * Computes matrix-matrix multiplication in case one is diagonal matrix
 *   - if isinv is 0
 *     res = a * d
 *   - if isinv is 1
 *     res = a * d^-1
 *
 * Input
 * -----
 * mat   : Matrix of size m x n
 * dmat  : Diagonal matrix of size n x n
 * isinv : 1 or 0
 *
 * Output
 * ------
 * res   : Matrix of size m x n
 */
AIA_API void aiatensor__(diagmm)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *dmat, bool isinv);

/**
 * Description
 * -----------
 * Computes the following
 *   res = a + alpha * d
 *
 * Input
 * -----
 * mat   : Matrix of size n x n
 * alpha : Multiplying factor
 * dmat  : Diagonal matrix of size n x n
 *
 * Output
 * ------
 * res   : Matrix of size n x n
 *
 */
AIA_API void aiatensor__(cadddiag)(AIATensor_ *res, AIATensor_ *mat, T alpha, AIATensor_ *dmat);

/**
 * Description
 * -----------
 * Computes inverse of a diagonal matrix
 *
 * Input
 * -----
 * mat    : Diagonal matrix of size n x n
 *
 * Output
 * ------
 * matinv : Inverse matrix of d
 */
AIA_API void aiatensor__(diaginv)(AIATensor_ *matinv, AIATensor_ *mat);

AIA_API T aiatensor__(xTAdiagx)(AIATensor_ *x, AIATensor_ *dmat);

/**
 * Description
 * -----------
 * Returns result of x.T * d * y where d is a diagonal matrix
 *
 * Input
 * -----
 * x    : Vector of size d
 * dmat : Diagonal matrix of size d x d
 * y    : Vector of size d
 *
 * Output
 * ------
 * Returns result of type T
 */
AIA_API T aiatensor__(xTAdiagy)(AIATensor_ *x, AIATensor_ *dmat, AIATensor_ *y);

AIA_API T aiatensor__(xTAdiagIx)(AIATensor_ *x, AIATensor_ *dmat);

AIA_API T aiatensor__(xTAdiagIy)(AIATensor_ *x, AIATensor_ *dmat, AIATensor_ *y);

#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor/diagmath.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_DIAGMATH_H
#endif