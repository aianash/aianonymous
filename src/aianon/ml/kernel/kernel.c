#include <aianon/ml/kernel/kernel.h>

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
 * lambda : Length scale vector of size d
 * alpha  : Signal variance of kernel
 *
 * Output
 * ------
 * K      : Kernel matrix of size n x m
 */
void aiakernel_rbf__(mpcreate)(AIATensor_ *K, AIATensor_ *X, AIATensor_ *Y, AIATensor_ *lambda, T alpha) {
  if(Y == NULL) Y = X;

  aia_argcheck(aiatensor__(nDimension)(X) == 2, 1, "X must be 2-dimensional");
  aia_argcheck(aiatensor__(nDimension)(Y) == 2, 2, "Y must be 2-dimensional");
  aia_argcheck(aiatensor__(size)(X, 1) == aiatensor__(size)(Y, 1), 2, "incosistent tensor sizes");

  long n = aiatensor__(size)(X, 0);
  long m = aiatensor__(size)(Y, 0);
  long d = aiatensor__(size)(X, 1);

  AIATensor_ *K_ = aiatensor__(newVector)(n * m);

  long lambda_stride = lambda->stride[0];
  T *lambda_data = aiatensor__(data)(lambda);

  AIA_TENSOR_CROSS_DIM_APPLY(T, X, T, Y, T, K_, 1,
                            T sum = 0;
                            int idx;
                            for(idx = 0; idx < d; idx++) {
                              sum += (pow(X_data[idx * X_stride] - Y_data[idx * Y_stride], 2) * lambda_data[idx * lambda_stride]);
                            }
                            sum /= -0.5;
                            *K__data = exp(sum) * pow(alpha, 2);
                            );
  aiatensor__(resize2d)(K_, n, m);
  aiatensor__(copy)(K, K_);
  aiatensor__(free)(K_);
}

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
void aiakernel_rbf__(sgcreate)(T *k, AIATensor_ *x, AIATensor_ *y, AIATensor_ *lambda, T alpha) {
  if(y == NULL) y = x;

  aia_argcheck(aiatensor__(isVector)(x), 2, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(y), 3, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(lambda), 4, "lambda should be 1-d tensor");
  aia_argcheck(x->size[0] == y->size[0], 2, "incosistent tensor size");
  aia_argcheck(x->size[0] == lambda->size[0], 4, "incosistent tensor size for lambda");

  AIATensor_ *diff = aiatensor__(empty)();

  aiatensor__(csub)(diff, x, 1, y);
  *k = aiatensor__(xTLy)(diff, lambda, diff);
  *k *= -0.5;
  *k = exp(*k) * pow(alpha, 2);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/kernel/kernel.c"
#include <aianon/core/erasure.h>