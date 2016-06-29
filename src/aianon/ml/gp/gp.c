#include <aianon/ml/gp/gp.h>

#ifdef ERASED_TYPE_PRESENT


/**
 * Description
 * -----------
 * Prediction for multiple certain input
 *
 * Input
 * -----
 * K     : Cholesky factorization of (Ktrain + sigma^2 * I) as obtained using potrf
 * Kx    : Kernel matrix of test datapoints
 * Kxx   : Cross kernel matrix size n x m where m is the number of test datapoints
 *          and n is number of training datapoints
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Covariance matrix of predictive posterior distribution
 */
void aiagp__(vpredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta) {
  long n = K->size[0];

  AIATensor_ *alpha  = aiatensor__(newVector)(n);
  AIATensor_ *KxTKKx = aiatensor__(empty)();
  AIATensor_ *KxT    = aiatensor__(empty)();

  aiatensor__(transpose)(KxT, Kx, 0, 1);
  aiatensor__(mv)(fmean, KxT, beta);

  aiatensor__(potrs)(alpha, Kx, K, uplo);
  aiatensor__(mv)(KxTKKx, KxT, alpha);
  aiatensor__(csub)(fcov, Kxx, 1, KxTKKx);

  aiatensor__(free)(alpha);
  aiatensor__(free)(KxTKKx);
}


/**
 * Descriotion
 * -----------
 * Prediction for single certain input
 *
 * Input
 * -----
 * K     : Cholesky factorization of (Ktrain + sigma^2 * I) as obtained using potrf
 * Kx    : Cross kernel Vector of size n
 * Kxx   : Kernel value for test input. Scalar of type T.
 *
 * Output
 * ------
 * fmean : Mean of predictive posterior distribution
 * fcov  : Standard deviation of predictive posterior distribution
 */
void aiagp__(soredc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, T Kxx, AIATensor_ *beta) {
  long n = K->size[0];

  AIATensor_ *alpha = aiatensor__(newVector)(n);
  AIATensor_ *KxT   = aiatensor__(empty)();

  aiatensor__(transpose)(KxT, Kx, 0, 1);

  *fmean = aiatensor__(dot)(KxT, beta);

  aiatensor__(potrs)(alpha, Kx, K, uplo);
  T KxTKKx = aiatensor__(dot)(KxT, alpha);

  *fcov = Kxx - KxTKKx;
}

/**
 * Description
 * -----------
 * Performs following computation
 * beta = (K + sigma^2 * I)^-1 * y
 *
 * Input
 * -----
 * K    : Cholesky factorization of (K + sigma^2 * I) as obtained using potrf
 * y    : vector of size n with training data
 * uplo : 'U' if A contains the uppar triangular matrix
 *        'L' if A contains the lower triangular matrix
 *
 * Output
 * ------
 * beta : vector of size n
 */
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