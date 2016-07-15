#include <aianon/ml/gp/gp.h>

#ifdef ERASED_TYPE_PRESENT

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// PRIVATE HELPER FUNCTIONS /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Description
 * -----------
 * Computes vector q with ith entry of q as
 *   q_i = alpha^2 * | xcov * lambda^-1 + I |^-1/2 * exp( -1/2 * (x_i - xxm)** * (xxcov + lambda)^-1 * (x_i - xxm) )
 *   where xm and xcov are mean and covariance matrix of test data respectively.
 *
 * Input
 * -----
 * X: Training input data. Matrix of size n x d.
 *
 * Output
 * ------
 * q: Vector of size
 */
static void aiagp__(calcq)(AIATensor_ *q, AIATensor_ *X, AIATensor_ *lambda, T alpha, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *Kxmu;
  T const_;
  AIATensor_ *constmat = aiatensor__(empty)();
  AIATensor_ *XxcovpL = aiatensor__(empty)();
  char *uplo = "L";

  aiatensor__(diagmm)(constmat, Xxcov, lambda, TRUE);
  aiatensor__(aIpX)(constmat, NULL, 1);
  const_ = pow(aiatensor__(detsymm)(constmat), -0.5);

  // compute cholesky of Xxcov + lambda
  aiatensor__(cadd)(XxcovpL, Xxcov, 1, lambda);
  aiatensor__(potrf)(XxcovpL, NULL, uplo);

  // compute exp part of q
  Kxmu = aiakernel_se__(matrix)(NULL, X, Xxm, alpha, XxcovpL, FALSE, uplo);
  aiatensor__(mul)(q, Kxmu, const_);

  aiatensor__(free)(constmat);
  aiatensor__(free)(Kxmu);
}

/**
 * Description
 * -----------
 * Computes Q part of covariance matrix for uncertain input case where Qij is given by
 *   Qij = |2 * sigma * lambda^-1|^-1/2 * exp(-1/2 * (Xxm - zij)** * (lambda/2 + sigma)^-1 * (Xxm - zij))
 *         * exp(-1/2 * (xi - xj)** * (2 * lambda)^-1 * (xi - xj))
 *
 * Input
 * -----
 * X      : Input data matrix of size n x d
 * lambda : Length scale matrix
 * Xxm    : Mean of test data
 * Xxcov  : Covariance matrix of test data
 *
 * Output
 * ------
 * Q      : Matrix of size n x n
 */
static void aiagp__(calcQ)(AIATensor_ *Q, AIATensor_ *X, AIATensor_ *lambda, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *K1, *K2, *constmat, *Y = X;
  T const_;
  long n = X->size[0], d = X->size[1];
  AIATensor_ *Z       = aiatensor__(empty)();
  AIATensor_ *lenscal = aiatensor__(new)(lambda);

  aiatensor__(resize2d)(Z, n * n, d);
  aiatensor__(diagmm)(constmat, Xxcov, lambda, TRUE);
  aiatensor__(aIpX)(constmat, NULL, 1);
  aiatensor__(mul)(constmat, constmat, 2);
  const_ = pow(aiatensor__(detsymm)(constmat), -0.5);

  aiatensor__(mul)(lenscal, lambda, 2);
  K1 = aiakernel_se__(matrix)(NULL, X, NULL, 1, lenscal, TRUE, NULL);

  AIA_TENSOR_CROSS_DIM_APPLY3(T, X, T, Y, T, Z, 0,
                              aiablas__(copy)(d, Y_data, Y_stride, Z_data, Z_stride);
                              aiablas__(axpy)(d, -1, X_data, X_stride, Z_data, Z_stride);
                              aiablas__(scal)(d, -1, Z_data, Z_stride);
                              );

  aiatensor__(cadddiag)(lenscal, Xxcov, 0.5, lambda);
  K2 = aiakernel_se__(matrix)(NULL, Z, Xxm, 1, lenscal, TRUE, NULL);
  aiatensor__(resize2d)(K2, n, n);

  AIA_TENSOR_APPLY3(T, K1, T, K2, T, Q, *Q_data = const_ * *K1_data * *K2_data;);

  aiatensor__(free)(K1);
  aiatensor__(free)(K2);
  aiatensor__(free)(Y);
  aiatensor__(free)(Z);
  aiatensor__(free)(lenscal);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// PUBLIC FUNCTIONS /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void aiagp__(vpredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta) {
  long n = K->size[0];
  AIATensor_ *KxT = aiatensor__(empty)();

  // calculation of fmean
  aiatensor__(transpose)(KxT, Kx, 0, 1);
  aiatensor__(mv)(fmean, KxT, beta);

  // calculation of fcov
  aiatensor__(XTAsymmIXpaY)(fcov, Kx, K, uplo, -1, Kxx);
  aiatensor__(mul)(fcov, fcov, -1);

  aiatensor__(free)(KxT);
}

void aiagp__(spredc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, T Kxx, AIATensor_ *beta) {
  long n = K->size[0];

  AIATensor_ *alpha = aiatensor__(newVector)(n);
  AIATensor_ *KxT   = aiatensor__(empty)();

  // calculation of fmean
  aiatensor__(transpose)(KxT, Kx, 0, 1);
  *fmean = aiatensor__(dot)(KxT, beta);

  // calculation of fcov
  T KxTKKx = aiatensor__(xTAsymmx)(Kx, K);
  *fcov = Kxx - KxTKKx;
}

void aiagp__(preduc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *lambda, T alpha, AIATensor_ *X, AIATensor_ *beta, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *q = aiatensor__(empty)();
  aiagp__(calcq)(q, X, lambda, alpha, Xxm, Xxcov);
  *fmean = aiatensor__(dot)(beta, q);

  AIATensor_ *Q = aiatensor__(empty)();
  aiagp__(calcQ)(Q, X, lambda, Xxm, Xxcov);
  AIATensor_ *KIQ = aiatensor__(empty)();
  aiatensor__(potrs)(KIQ, Q, K, uplo);
  *fcov = pow(alpha, 2) - aiatensor__(trace)(KIQ) + aiatensor__(xTAx)(beta, Q) - pow(*fmean, 2);
}

void aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *K, const char* uplo, AIATensor_ *y) {
  aia_argcheck(K->nDimension == 2, 1, "K should be 2-dimensional matrix");
  aia_argcheck(y->nDimension == 1, 2, "y should be a vector");
  aia_argcheck(K->size[0] == y->size[0], 2, "inconsistent tensor size");

  aiatensor__(potrs)(beta, y, K, uplo);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/gp/gp.c"
#include <aianon/core/erasure.h>