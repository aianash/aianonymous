#include <aianon/ml/kernel/kernel.h>

#ifdef ERASED_TYPE_PRESENT

void aiakernel_rbf__(mpcreate)(AIATensor_ *K, AIATensor_ *X, AIATensor_ *Y, T alpha, AIATensor_ *lambda, bool isdiag) {
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

  if(isdiag) {
    AIA_TENSOR_CROSS_DIM_APPLY(T, X, T, Y, T, K_, 1,
                              T sum = 0;
                              int idx;
                              for(idx = 0; idx < d; idx++) {
                                sum += (pow(X_data[idx * X_stride] - Y_data[idx * Y_stride], 2) * lambda_data[idx * lambda_stride]);
                              }
                              sum *= -0.5;
                              *K__data = exp(sum) * pow(alpha, 2);
                              );
  } else {
    AIATensor_ *diff = aiatensor__(newVector)(d);
    AIATensor_ *y = aiatensor__(newVector)(d);
    T *diff_data = aiatensor__(data)(diff);
    long diff_stride = diff->stride[0];
    AIA_TENSOR_CROSS_DIM_APPLY(T, X, T, Y, T, K_, 1,
                              aiablas__(copy)(d, Y_data, Y_stride, diff_data, diff_stride);
                              aiablas__(axpy)(d, -1, X_data, X_stride, diff_data, diff_stride);
                              aiatensor__(trtrs)(y, diff, lambda, "L", "N", "N");
                              *K__data = aiatensor__(dot)(y, y);
                              *K__data *= -0.5;
                              *K__data = exp(*K__data) * pow(alpha, 2);
                              );
    aiatensor__(free)(diff);
    aiatensor__(free)(y);
  }
  aiatensor__(resize2d)(K_, n, m);
  aiatensor__(freeCopyTo)(K, K_);
}

void aiakernel_rbf__(sgcreate)(T *k, AIATensor_ *x, AIATensor_ *y, T alpha, AIATensor_ *lambda, int isdiag) {
  if(y == NULL) y = x;

  aia_argcheck(aiatensor__(isVector)(x), 2, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(y), 3, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(lambda), 4, "lambda should be 1-d tensor");
  aia_argcheck(x->size[0] == y->size[0], 2, "incosistent tensor size");
  aia_argcheck(x->size[0] == lambda->size[0], 4, "incosistent tensor size for lambda");

  AIATensor_ *diff = aiatensor__(empty)();
  aiatensor__(csub)(diff, x, 1, y);

  if(isdiag) {
    *k = aiatensor__(xtdy)(diff, lambda, diff, true);
  } else {
    *k = aiatensor__(xtay)(diff, lambda, diff, true);
  }
  *k *= -0.5;
  *k = exp(*k) * pow(alpha, 2);
  aiatensor__(free)(diff);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/kernel/kernel.c"
#include <aianon/core/erasure.h>