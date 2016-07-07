#include <aianon/ml/gp/gp.h>

#ifdef ERASED_TYPE_PRESENT

void aiagp__(vpredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta) {
  long n = K->size[0];
  AIATensor_ *KxT = aiatensor__(empty)();

  // calculation of fmean
  aiatensor__(transpose)(KxT, Kx, 0, 1);
  aiatensor__(mv)(fmean, KxT, beta);

  // calculation of fcov
  aiatensor__(XTAsymmIXpaY)(fcov, Kx, -1, Kxx, K);
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

void aiagp__(vpreduc)() {}

void aiagp__(spreduc)(T *fmean, T *fcov, AIATensor_ *K, AIATensor_ *lambda, AIATensor_ *X, AIATensor_ *beta, AIATensor_ *xm, AIATensor_ *xcov) {}

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
void aiagp__(calcq)(AIATensor_ *q, AIATensor_ *X, AIATensor_ *lambda, T alpha, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *Kxmu;
  T const_;
  AIATensor_ *constmat = aiatensor__(empty)();
  AIATensor_ *XxcovpL = aiatensor__(empty)();
  char *uplo = "L";

  aiatensor__(diagmm)(constmat, Xxcov, lambda, TRUE);
  aiatensor__(aIpx)(constmat, constmat, 1);
  const_ = pow(aiatensor__(det)(constmat), -0.5);

  // compute cholskey of Xxcov + lambda
  aiatensor__(cadd)(XxcovpL, Xxcov, 1, lambda);
  aiatensor__(potrf)(XxcovpL, NULL, uplo);

  // compute exp part of q
  Kxmu = aiakernel_se__(matrix)(NULL, X, Xxm, alpha, XxcovpL, FALSE, uplo);
  aiatensor__(cmul)(q, Kxmu, const_);

  aiatensor__(free)(constmat);
  aiatensor__(free)(Kxmu);
}

void aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *K, AIATensor_ *y, const char* uplo) {
  aia_argcheck(K->nDimension == 2, 1, "K should be 2-dimensional matrix");
  aia_argcheck(y->nDimension == 1, 2, "y should be a vector");
  aia_argcheck(K->size[0] == y->size[0], 2, "inconsistent tensor size");

  aiatensor__(potrs)(beta, y, K, uplo);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/gp/gp.c"
#include <aianon/core/erasure.h>