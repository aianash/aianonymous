#ifndef AIA_ML_KERNEL_H
#define AIA_ML_KERNEL_H

#include <aianon/core/util.h>
#include <aianon/core/util/memory.h>
#include <aianon/core/math/blas.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/dimcrossapply.h>

#ifdef ERASED_TYPE_PRESENT

/**
 * Description
 * -----------
 * Returns RBF kernel matrix
 * Radial basis function (RBF) is given by
 *   k(x_i, y_j) = alpha^2 * exp( -1/2 * (x_i - y_j)** * lambda^-1 * (x_i - y_j) )
 *
 * Input
 * -----
 * X      : Matrix of size n x d where n is number of data points
 * Y      : Matrix of size m x d where m is number of data points
 * alpha  : Signal variance of kernel
 * lambda : Length scale vector of size d
 * isdiag : True if lambda is diagonal matrix, false otherwise
 *
 * Output
 * ------
 * K      : Kernel matrix of size n x m
 */
void aiakernel_rbf__(mpcreate)(AIATensor_ *K, AIATensor_ *X, AIATensor_ *Y, T alpha, AIATensor_ *lambda, bool isdiag);

/**
 * Description
 * -----------
 * Return RBF kernel function value for two data points given by
 *   k(x, y) = alpha^2 * exp( -1/2 * (x - y)** * lambda^-1 * (x - y) )
 *
 * Input
 * -----
 * x      : Vector of size d
 * y      : Vector of size d
 * lambda : Length scale vector of size d
 * alpha  : Signal variance of kernel
 *
 * Output
 * ------
 * k      : Scalar of type T
 */
AIA_API void aiakernel_rbf__(sgcreate)(T *k, AIATensor_ *x, AIATensor_ *y, AIATensor_ *lambda, T alpha);

#endif

#ifndef aiakernel_rbf_
#define aiakernel_rbf_(type, name) AIA_FN_ERASE_(kernel_rbf, type, name)
#define aiakernel_rbf__(name) aiakernel_rbf_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/kernel/kernel.h"
#include <aianon/core/erasure.h>

#endif