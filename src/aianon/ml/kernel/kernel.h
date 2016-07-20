#ifndef AIA_ML_KERNEL_H

#include <aianon/core/util.h>
#include <aianon/core/util/memory.h>
#include <aianon/core/math/blas.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/math.h>
#include <aianon/tensor/diagmath.h>
#include <aianon/tensor/linalg.h>
#include <aianon/tensor/dimcrossapply.h>

#ifdef ERASED_TYPE_PRESENT

/**
 * Description
 * -----------
 * Returns RBF kernel matrix
 * Radial basis function (RBF) is given by
 *   k(x_i, y_j) = alpha^2 * exp( -1/2 * (x_i - y_j).T * lambda^-1 * (x_i - y_j) )
 *
 * Input
 * -----
 * X      : Matrix of size n x d where n is number of data points
 * Y      : Matrix of size m x d where m is number of data points
 * alpha  : Signal variance of kernel
 * lambda : For a positive definite length scale matrix of size of size d x d
 *            - if isdiag is true, matrix should be diagonal
 *            - if isdiag is false, it should be cholesky decomposition of length scale matrix
 * isdiag : True if lambda is diagonal matrix, false otherwise
 * uplo   : "U" or "L" depending on whether lambda has upper or lower triangular matrix
 *          NULL if isdiad is TRUE
 *
 * Output
 * ------
 * K      : Kernel matrix of size n x m
 * If K is NULL, it creates a matrix and returns it. Client has to free this memory.
 *
 */
AIA_API AIATensor_ *aiakernel_se__(matrix)(AIATensor_ *K, AIATensor_ *X, AIATensor_ *Y, T alpha, AIATensor_ *lambda, bool isdiag, const char *uplo);

/**
 * Description
 * -----------
 * Return RBF kernel function value for two data points given by
 *   k(x, y) = alpha^2 * exp( -1/2 * (x - y).T * lambda^-1 * (x - y) )
 *
 * Input
 * -----
 * x      : Vector of size d
 * y      : Vector of size d
 * alpha  : Signal variance of kernel
 * lambda : For a positive definite length scale matrix of size of size d x d
 *            - if isdiag is true, matrix should be diagonal and should be length scale matrix
 *            - if isdiag is false, it should be cholesky decomposition of length scale matrix
 * isdiag : True if lambda is diagonal matrix, false otherwise
 * uplo   : "U" or "L" depending on whether lambda has upper or lower triangular matrix
 *          NULL if isdiad is TRUE
 * Output
 * ------
 * Returns Scalar of type T
 */
AIA_API T aiakernel_se__(value)(AIATensor_ *x, AIATensor_ *y, T alpha, AIATensor_ *lambda, int isdiag, const char *uplo);

#endif

#ifndef aiakernel_se_
#define aiakernel_se_(type, name) AIA_FN_ERASE_(kernel_se, type, name)
#define aiakernel_se__(name) aiakernel_se_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/kernel/kernel.h"
#include <aianon/core/erasure.h>

#define AIA_ML_KERNEL_H
#endif