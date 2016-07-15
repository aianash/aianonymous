#ifndef AIA_ML_GP_H

#include <aianon/core/util.h>
#include <aianon/core/math/blas.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/math.h>
#include <aianon/tensor/dimapply.h>
#include <aianon/tensor/diagmath.h>
#include <aianon/ml/kernel/kernel.h>

#ifdef ERASED_TYPE_PRESENT

/**
 * Description
 * -----------
 * Performs following computation
 *   beta = (K + sigma^2 * I)^-1 * y
 *
 * Input
 * -----
 * K    : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *        where sigma is noise variance
 * uplo : 'U' if K contains the uppar triangular matrix
 *        'L' if K contains the lower triangular matrix
 * y    : vector of size n with training data
 *
 * Output
 * ------
 * beta : vector of size n
 */
AIA_API void aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *K, const char* uplo, AIATensor_ *y);

/**
 * Description
 * -----------
 * Prediction for multiple certain input
 *
 * Input
 * -----
 * K     : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * Kx    : Kernel matrix of test datapoints
 * Kxx   : Cross kernel matrix size n x m where m is the number of test datapoints
 *          and n is number of training datapoints
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Covariance matrix of predictive posterior distribution
 */
AIA_API void aiagp__(vpredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta);

/**
 * Descriotion
 * -----------
 * Prediction for single certain input
 *
 * Input
 * -----
 * K     : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * uplo  : 'U' if K contains the uppar triangular matrix
 *         'L' if K contains the lower triangular matrix
 * Kx    : Cross kernel Vector of size n
 * Kxx   : Kernel value for test input. Scalar of type T.
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Standard deviation of predictive posterior distribution
 */
AIA_API void aiagp__(spredc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, T Kxx, AIATensor_ *beta);

/**
 * Desciption
 * ----------
 * Predicts mean and variance for uncertain test input. Works for only one test input.
 *
 * Input
 * -----
 * K      : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 * uplo   : 'U' if K contains the uppar triangular matrix
 *          'L' if K contains the lower triangular matrix
 * lambda : Length scale factor
 * alpha  : Signal variance of kernel
 * X      : Input data matrix of size n x d
 * beta   : As calculated using aiagp__(calcbeta) function
 * Xxm    : Mean of test input
 * Xxcov  : Covariance matrix of test input
 *
 * Output
 * ------
 * fmean  : Mean of output for test input
 * fcov   : Standard deviation of output for test input
 */
AIA_API void aiagp__(preduc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *lambda, T alpha, AIATensor_ *X, AIATensor_ *beta, AIATensor_ *Xxm, AIATensor_ *Xxcov);

#endif

#ifndef aiagp_
#define aiagp_(type, name) AIA_FN_ERASE_(gp, type, name)
#define aiagp__(name) aiagp_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/gp/gp.h"
#include <aianon/core/erasure.h>

#define AIA_ML_GP_H
#endif