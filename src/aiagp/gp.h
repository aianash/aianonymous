#ifndef AIA_ML_GP_H

#include <aiautil/util.h>
#include <aiatensor/blas.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>
#include <aiatensor/dimapply.h>
#include <aiatensor/diagmath.h>
#include <aiakernel/kernel.h>
#include <aiaoptim/optim.h>
#include <aiautil/math.h>

#ifdef ERASED_TYPE_PRESENT

typedef struct GPState_ {
  AIATensor_ *X;
  AIATensor_ *y;
  bool isokernel;
} GPState_;

/**
 * Description
 * -----------
 * Performs following computation
 *   beta = (K + sigma^2 * I)^-1 * y
 *
 * Input
 * -----
 * Kchol : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * mtype : UPPER_MAT or LOWER_MAT
 * y     : vector of size n with training data
 *
 * Output
 * ------
 * beta  : vector of size n
 *         when NULL, a new vector is created
 */
AIA_API AIATensor_ *aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *y);

/**
 * Description
 * -----------
 * Computes K1 matrix where (i, j)th entry is given by
 *   K1_ij = exp(-1/2 * (x_i - x_j).T * (2 * lambda)^-1 * (x_i - x_j))
 *
 * Input
 * -----
 * X      : Input data matrix of size n x d where n is number of samples
 * lambda : Diagonal matrix of length scale
 *
 * Output
 * ------
 * K1     : n x n matrix
 */
AIA_API void aiagp__(calcK1)(AIATensor_ *K1, AIATensor_ *X, AIATensor_ *lambda);

/**
 * Description
 * -----------
 * Prediction for multiple certain input
 *
 * Input
 * -----
 * Kchol : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * mtype : UPPER_MAT OR LOWER_MAT
 * Kx    : Cross kernel matrix size n x m where m is the number of test datapoints
 *         and n is number of training datapoints
 * Kxx   : Kernel matrix of test datapoints
 * beta  : As calculated using aiagp__(calcbeta) function
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Covariance matrix of predictive posterior distribution
 */
AIA_API void aiagp__(npredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta);

/**
 * Descriotion
 * -----------
 * Prediction for single certain input
 *
 * Input
 * -----
 * Kchol : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * mtype : UPPER_MAT OR LOWER_MAT
 * Kx    : Cross kernel Vector of size n
 * Kxx   : Kernel value for test input. Scalar of type T
 * beta  : As calculated using aiagp__(calcbeta) function
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Standard deviation of predictive posterior distribution
 */
AIA_API void aiagp__(spredc)(T *fmean, T *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *Kx, T Kxx, AIATensor_ *beta);

/**
 * Desciption
 * ----------
 * Predicts mean and variance for uncertain test input. Works for only one test input.
 *
 * Input
 * -----
 * Kchol  : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 * mtype  : UPPER_MAT OR LOWER_MAT
 * lambda : Length scale factor
 * alpha  : Signal variance of kernel
 * X      : Input data matrix of size n x d
 * beta   : As calculated using aiagp__(calcbeta) function
 * K1     : As calculated using aiagp__(calcK1) function
 * Xxm    : Mean of test input
 * Xxcov  : Covariance matrix of test input
 *
 * Output
 * ------
 * fmean  : Mean of output for test input
 * fcov   : Standard deviation of output for test input
 */
AIA_API void aiagp__(spreduc)(T *fmean, T *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *lambda, T alpha, AIATensor_ *X, AIATensor_ *beta, AIATensor_ *K1, AIATensor_ *Xxm, AIATensor_ *Xxcov);

/**
 * Desciption
 * ----------
 * Calculate evidence likelihood and derivative of likelihood wrt given parameters
 *
 * Input
 * -----
 * x      : vector of parameters [sigma, alpha, lambda1, lambda2, ...]
 * ops    : flag which tells what needs to be calculated.
 *          value, grad or both
 * state  : state of gp optimization
 *
 * Output
 * ------
 * fx     : value of function to be optimized
 * df_dx   : gradient of function to be optimized
 */
AIA_API void aiagp__(opfuncse)(AIATensor_ *x, T *fx, AIATensor_ *df_dx, opfunc_ops ops, void *state);

#endif

#ifndef aiagp_
#define GPState(type) AIA_STRUCT_ERASE_(type, gp)
#define aiagp_(type, name) AIA_FN_ERASE_(gp, type, name)

#define GPState_ GPState(T_)
#define aiagp__(name) aiagp_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiagp/gp.h"
#include <aiautil/erasure.h>

#define AIA_ML_GP_H
#endif