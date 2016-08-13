#include <aiagp/gp.h>

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

  aiatensor__(diagmm)(constmat, Xxcov, lambda, TRUE);
  aiatensor__(aEyepX)(constmat, NULL, 1);
  const_ = pow(aiatensor__(detpd)(constmat), -0.5);

  // compute cholesky of Xxcov + lambda
  aiatensor__(cadd)(XxcovpL, Xxcov, 1, lambda);
  aiatensor__(potrf)(XxcovpL, NULL, LOWER_MAT);

  // compute exp part of q
  Kxmu = aiakernel_se__(matrix)(NULL, X, Xxm, alpha, XxcovpL, LOWER_MAT);
  aiatensor__(mul)(q, Kxmu, const_);

  aiatensor__(free)(constmat);
  aiatensor__(free)(Kxmu);
}

/**
 * Description
 * -----------
 * Computes Q part of covariance matrix for uncertain input case where Qij is given by
 *   Qij = |2 * sigma * lambda^-1|^-1/2 * exp(-1/2 * (Xxm - zij).T * (lambda/2 + sigma)^-1 * (Xxm - zij)) * K1
 *   where
 *   K1 = exp(-1/2 * (xi - xj).T * (2 * lambda)^-1 * (xi - xj))
 *   K1 can be calculated using aiagp__(calcK1)
 *
 * Input
 * -----
 * X      : Input data matrix of size n x d
 * lambda : Length scale matrix
 * K1     : As calculated using aiagp__(calcK1)
 * Xxm    : Mean of test data
 * Xxcov  : Covariance matrix of test data
 *
 * Output
 * ------
 * Q      : Matrix of size n x n
 */
static void aiagp__(calcQ)(AIATensor_ *Q, AIATensor_ *X, AIATensor_ *lambda, AIATensor_ *K1, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *K2, *constmat, *Y = X;
  T const_;
  long n = X->size[0], d = X->size[1];
  AIATensor_ *Z       = aiatensor__(empty)();
  AIATensor_ *lenscal = aiatensor__(new)(lambda);

  aiatensor__(resize2d)(Z, n * n, d);
  aiatensor__(diagmm)(constmat, Xxcov, lambda, TRUE);
  aiatensor__(aEyepX)(constmat, NULL, 1);
  aiatensor__(mul)(constmat, constmat, 2);
  const_ = pow(aiatensor__(detpd)(constmat), -0.5);

  aiatensor__(mul)(lenscal, lambda, 2);
  AIA_TENSOR_CROSS_DIM_APPLY3(T, X, T, Y, T, Z, 0,
                              aiablas__(copy)(d, Y_data, Y_stride, Z_data, Z_stride);
                              aiablas__(axpy)(d, -1, X_data, X_stride, Z_data, Z_stride);
                              aiablas__(scal)(d, -1, Z_data, Z_stride);
                              );

  aiatensor__(cadddiag)(lenscal, Xxcov, 0.5, lambda);
  K2 = aiakernel_se__(matrix)(NULL, Z, Xxm, 1, lenscal, DIAG_MAT);
  aiatensor__(resize2d)(K2, n, n);

  AIA_TENSOR_APPLY3(T, K1, T, K2, T, Q, *Q_data = const_ * *K1_data * *K2_data;);

  aiatensor__(free)(K1);
  aiatensor__(free)(K2);
  aiatensor__(free)(Y);
  aiatensor__(free)(Z);
  aiatensor__(free)(lenscal);
}

/**
 * Description
 * -----------
 * Performs following computation
 *   gamma = (K + sigma^2 * I)^-1 -  (K + sigma^2 * I)^-1 * y * yT * (K + sigma^2 * I)^-1
 *
 * Input
 * -----
 * Kchol : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 *         where sigma is noise variance
 * mtype : DIAG_MAT or UPPER_MAT or LOWER_MAT
 * beta  : As calculated using aiagp__(calcbeta) function (K + sigma^2 * I)^-1 * y)
 *
 * Output
 * ------
 * gamma   : matrix of size n x n
 */
static void aiagp__(calcgamma)(AIATensor_ *gamma, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *beta) {
  aiatensor__(resizeAs)(gamma, Kchol);
  // calculate beta * betaT
  aiatensor__(xxT)(gamma, beta);
  // calculate (K + sigma ^ 2 * I) ^ -1
  AIATensor_ *inv = aiatensor__(empty)();
  AIATensor_ *eye = aiatensor__(empty)();
  // Identity matrix
  aiatensor__(zeros)(eye, 2, Kchol->size, NULL);
  aiatensor__(aEyepX)(eye, NULL, 1);
  aiatensor__(potrs)(inv, eye, Kchol, mtype);
  aiatensor__(csub)(gamma, inv, 1, gamma);

  aiatensor__(free)(inv);
  aiatensor__(free)(eye);
}

/**
 * Description
 * -----------
 * Calculate negative log evidence
 *
 * Input
 * -----
 * d       : number of dimensions in multivariate gaussian
 * KPSchol : cholskey decomposition of (K + sigma ^ 2 * I)
 * y       : vector of size n with training data outputs
 * beta    : As calculated using aiagp__(calcbeta) function (K + sigma^2 * I)^-1 * y)
 *
 * Output
 * ------
 * res   : negative log evidence for given parameters
 */
static void aiagp__(calcnle)(T *res, long d, AIATensor_ *KPSchol, AIATensor_ *y, AIATensor_ *beta) {
  T *KPSchol_data;
  T detkps;
  long idx;

  // calculate determinant of (K + sigma ^ 2 * I)
  // not using tensor detpd as chol is already calculated
  KPSchol_data = aiatensor__(data)(KPSchol);
  detkps = 1;
  for(idx = 0; idx < KPSchol->size[0]; idx++) {
    detkps *= KPSchol_data[idx * (KPSchol->stride[0] + KPSchol->stride[1])];
  }
  detkps = pow(detkps, 2);

  *res = 0.5 * aiatensor__(dot)(beta, y) + 0.5 * log(detkps) + 0.5 * d * log(2 * PI);
}

/**
 * Description
 * -----------
 * Calculate gradient of negative log evidence
 *
 * Input
 * -----
 * K         : covariance matrix of trainin inputs
 * KPSchol   : cholskey decomposition of (K + sigma ^ 2 * I)
 * beta      : As calculated using aiagp__(calcbeta) function (K + sigma^2 * I)^-1 * y)
 * X         : training input matrix
 * lambda    : length scale vector (can be sinle value in case of isotropic)
 * sigma     : value of parameter sigma
 * alpha     : value of parameter alpha
 * isokernel : boolean which tells kernel length scale is isotropic or not
 *
 * Output
 * ------
 * res   : vector of gradient values of nle
 */
static void aiagp__(calcdnle)(AIATensor_ *res, AIATensor_ *K, AIATensor_ *KPSchol, AIATensor_ *beta,
                              AIATensor_ *X, AIATensor_ *lambda, T sigma, T alpha, bool isokernel) {
  AIATensor_ *gamma = aiatensor__(empty)();
  AIATensor_ *gammaT = aiatensor__(empty)();
  AIATensor_ *res_ = aiatensor__(empty)();

  T *lambda_data, *X_data, *res__data;
  T dalpha;

  long lambda_stride, d, n, ridx, cidx, idx;

  aiatensor__(resize1d)(res_, lambda->size[0] + 2);
  aiatensor__(fill)(res_, 0);
  res__data = aiatensor__(data)(res_);

  // calculate gamma
  aiagp__(calcgamma)(gamma, KPSchol, LOWER_MAT, beta);
  // get transpose of gamma
  aiatensor__(transpose)(gammaT, gamma, 0, 1);

  // calculate gradient of nle wrt sigma
  res__data[0] = sigma * aiatensor__(trace)(gamma);

  // calculate gradient of nle wrt alpha
  dalpha = 0;
  AIA_TENSOR_APPLY2(T, K, T, gammaT, dalpha += se_grad_alpha(*K_data, alpha) * *gammaT_data;);
  res__data[res_->stride[0]] = 0.5 * dalpha;

  // calculate gradient of nle wrt length scale
  lambda_stride = lambda->stride[0];
  lambda_data = aiatensor__(data)(lambda);
  X_data = aiatensor__(data)(X);
  d = X->size[1];
  n = X->size[0];
  ridx = 0;
  cidx = 0;

  if(isokernel) {
    AIA_TENSOR_APPLY2(T, K, T, gammaT,
                    long didx;
                    T xdifsq;
                    for(didx = 0; didx < d; didx++) {
                      // squared distance over all dimensioans
                      xdifsq += pow(X_data[ridx * X->stride[0] + didx * X->stride[1]] - X_data[cidx * X->stride[0] + didx * X->stride[1]], 2);
                    }
                    res__data[2 * res_->stride[0]] += se_grad_lambda(*K_data, xdifsq ,lambda_data[didx * lambda_stride]) * *gammaT_data;
                    cidx += 1;
                    if(cidx % n == 0) {
                      ridx += 1;
                      cidx = 0;
                    });
    res__data[2 * res_->stride[0]] *= lambda_data[0];
  } else {
    AIA_TENSOR_APPLY2(T, K, T, gammaT,
                    long didx;
                    T xdifsq;

                    for(didx = 0; didx < d; didx++) {
                      // squared distace over a single dimension
                      xdifsq = pow(X_data[ridx * X->stride[0] + didx * X->stride[1]] - X_data[cidx * X->stride[0] + didx * X->stride[1]], 2);
                      res__data[(didx + 2) * res_->stride[0]] += se_grad_lambda(*K_data, xdifsq ,lambda_data[didx * lambda_stride]) * *gammaT_data;
                    }
                    cidx += 1;
                    if(cidx % n == 0) {
                      ridx += 1;
                      cidx = 0;
                    });
    for (idx = 0; idx < d; idx++) {
      res__data[(idx + 2) * res_->stride[0]] *= lambda_data[idx * lambda->stride[0]];
    }
  }

  aiatensor__(resizeAs)(res, res_);
  aiatensor__(freeCopyTo)(res_, res);

  aiatensor__(free)(gamma);
  aiatensor__(free)(gammaT);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// PUBLIC FUNCTIONS /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void aiagp__(npredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta) {
  long n = Kchol->size[0];
  AIATensor_ *KxT = aiatensor__(empty)();

  // resize fmean and fcov
  aiatensor__(resize1d)(fmean, Kxx->size[0]);
  aiatensor__(resize2d)(fcov, Kxx->size[0], Kxx->size[1]);

  // calculation of fmean
  aiatensor__(transpose)(KxT, Kx, 0, 1);
  aiatensor__(mv)(fmean, KxT, beta);

  // calculation of fcov
  aiatensor__(XTApdIXpaY)(fcov, Kx, Kchol, mtype, -1, Kxx);
  aiatensor__(mul)(fcov, fcov, -1);

  aiatensor__(free)(KxT);
}

void aiagp__(spredc)(T *fmean, T *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *Kx, T Kxx, AIATensor_ *beta) {
  long n = Kchol->size[0];

  // calculation of fmean
  *fmean = aiatensor__(dot)(Kx, beta);

  // calculation of fcov
  T KxTKKx = aiatensor__(xTApdIx)(Kx, Kchol, mtype);
  *fcov = Kxx - KxTKKx;
}

void aiagp__(spreduc)(T *fmean, T *fcov, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *lambda, T alpha, AIATensor_ *X, AIATensor_ *beta, AIATensor_ *K1, AIATensor_ *Xxm, AIATensor_ *Xxcov) {
  AIATensor_ *q = aiatensor__(empty)();

  aiagp__(calcq)(q, X, lambda, alpha, Xxm, Xxcov);
  *fmean = aiatensor__(dot)(beta, q);

  AIATensor_ *Q = aiatensor__(empty)();
  aiagp__(calcQ)(Q, X, lambda, K1, Xxm, Xxcov);
  AIATensor_ *KIQ = aiatensor__(empty)();
  aiatensor__(potrs)(KIQ, Q, Kchol, mtype);
  *fcov = pow(alpha, 2) - aiatensor__(trace)(KIQ) + aiatensor__(xTAx)(beta, Q) - pow(*fmean, 2);
}

AIATensor_ *aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *Kchol, MatrixType mtype, AIATensor_ *y) {
  if(beta == NULL) beta = aiatensor__(newCopy)(y);

  aia_argcheck(aiatensor__(isMatrix)(Kchol), 1, "Kchol should be 2-dimensional matrix");
  aia_argcheck(aiatensor__(isVector)(y), 2, "y should be a vector");
  aia_argcheck(Kchol->size[0] == y->size[0], 2, "inconsistent tensor size");

  aiatensor__(resize1d)(beta, Kchol->size[0]);
  aiatensor__(copy)(beta, y);
  aiatensor__(potrs)(beta, y, Kchol, mtype);
  return beta;
}

void aiagp__(calcK1)(AIATensor_ *K1, AIATensor_ *X, AIATensor_ *lambda) {
  aia_argcheck(aiatensor__(isMatrix)(X), 2, "X should be 2-dimensional matrix");
  aia_argcheck(aiatensor__(isVector)(lambda), 3, "lambda should be a diagonal matrix");

  AIATensor_ *lam2 = aiatensor__(newCopy)(lambda);

  aiatensor__(mul)(lam2, lambda, 2);
  aiakernel_se__(matrix)(K1, X, NULL, 1, lam2, DIAG_MAT);

  aiatensor__(free)(lam2);
}

void aiagp__(opfuncse)(AIATensor_ *x, T *fx, AIATensor_ *df_dx, opfunc_ops ops, void *state) {
  AIAGpState_ *gpstate = (AIAGpState_ *)state;
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be a vector");
  aia_argcheck(ops >= ONLY_F && ops <= F_N_GRAD, 4, "Invalid option for GP optimize function");
  if (gpstate->isokernel) {
    aia_argcheck(x->size[0] == 3, 1, "x vector should of size 3 for isotropic kernel. [sigma, alpha, lambda]");
  } else {
    aia_argcheck(gpstate->X->size[1] == x->size[0] - 2, 1, "kernel length scale dimension and "
      "input dimension should be equal in case of anisotropic kernel");
  }

  AIATensor_ *lambda = aiatensor__(empty)();
  AIATensor_ *K = aiatensor__(empty)();
  AIATensor_ *KPS = aiatensor__(empty)();
  AIATensor_ *KPSchol = aiatensor__(empty)();
  AIATensor_ *beta = aiatensor__(empty)();

  T *x_data, *ele;
  T sigma, alpha;

  // get new lambda, sigma and alpha
  x_data = aiatensor__(data)(x);
  sigma = x_data[0];
  alpha = x_data[x->stride[0]];
  aiatensor__(narrow)(lambda, x, 0, 2, x->size[0] - 2);
  lambda = aiatensor__(newCopy)(lambda);
  // use squared lambda as length scale
  ele = NULL;
  vforeach(ele, lambda) {
    *ele = *ele * *ele;
  }
  endvforeach()

  // calculate covariance matrix for input X
  aiakernel_se__(matrix)(K, gpstate->X, NULL, alpha, lambda, DIAG_MAT);
  // add noise variance
  aiatensor__(aEyepX)(KPS, K, sigma * sigma);
  // calculate cholskey decomposition of KPS
  aiatensor__(resizeAs)(KPSchol, KPS);
  aiatensor__(potrf)(KPSchol, KPS, LOWER_MAT);
  // calculate beta
  aiagp__(calcbeta)(beta, KPSchol, LOWER_MAT, gpstate->y);

  switch (ops) {
    case ONLY_F:
      aiagp__(calcnle)(fx, gpstate->X->size[1], KPSchol, gpstate->y, beta);
      break;
    case ONLY_GRAD:
      aiagp__(calcdnle)(df_dx, K, KPSchol, beta, gpstate->X, lambda, sigma, alpha, gpstate->isokernel);
      break;
    case F_N_GRAD:
      aiagp__(calcnle)(fx, gpstate->X->size[1], KPSchol, gpstate->y, beta);
      aiagp__(calcdnle)(df_dx, K, KPSchol, beta, gpstate->X, lambda, sigma, alpha, gpstate->isokernel);
      break;
    default:
      aia_error("Invalid option for GP optimize function");
      break;
  }

  aiatensor__(free)(lambda);
  aiatensor__(free)(K);
  aiatensor__(free)(KPS);
  aiatensor__(free)(KPSchol);
  aiatensor__(free)(beta);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiagp/gp.c"
#include <aiautil/erasure.h>