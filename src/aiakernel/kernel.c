#include <aiakernel/kernel.h>

#ifdef ERASED_TYPE_PRESENT

AIATensor_ *aiakernel_se__(matrix)(AIATensor_ *K, AIATensor_ *X, AIATensor_ *Y, T alpha, AIATensor_ *lambda, bool isdiag, const char *uplo) {
  if(K == NULL) K = aiatensor__(empty)();
  if(Y == NULL) Y = X;

  aia_argcheck(aiatensor__(isMatrix)(X), 1, "X must be 2-dimensional");
  aia_argcheck(aiatensor__(isMatrix)(Y), 2, "Y must be 2-dimensional");
  aia_argcheck(X->size[1] == Y->size[1], 2, "incosistent tensor sizes");

  long n = X->size[0];
  long m = Y->size[0];
  long d = X->size[1];


  aiatensor__(resize2d)(K, X->size[0], Y->size[0]);
  AIATensor_ *K_ = aiatensor__(emptyVector)(n * m);

  long lambda_stride = lambda->stride[0];
  T *lambda_data = aiatensor__(data)(lambda);

  if(isdiag) {
    AIA_TENSOR_CROSS_DIM_APPLY3(T, X, T, Y, T, K_, 1,
                              T sum = 0;
                              int idx;
                              for(idx = 0; idx < d; idx++) {
                                sum += (pow(X_data[idx * X_stride] - Y_data[idx * Y_stride], 2) / lambda_data[idx * lambda_stride]);
                              }
                              sum *= -0.5;
                              *K__data = exp(sum) * pow(alpha, 2);
                              );
  } else {
    AIATensor_ *diff = aiatensor__(emptyVector)(d);
    AIATensor_ *y    = aiatensor__(emptyVector)(d);
    AIATensor_ *tmp  = aiatensor__(newCopy)(lambda);
    T *diff_data     = aiatensor__(data)(diff);
    long diff_stride = diff->stride[0];
    AIA_TENSOR_CROSS_DIM_APPLY3(T, X, T, Y, T, K_, 1,
                              aiablas__(copy)(d, Y_data, Y_stride, diff_data, diff_stride);
                              aiablas__(axpy)(d, -1, X_data, X_stride, diff_data, diff_stride);
                              aiatensor__(trtrs)(tmp, y, diff, lambda, uplo, "N", "N");
                              *K__data = aiatensor__(dot)(y, y);
                              *K__data *= -0.5;
                              *K__data = exp(*K__data) * pow(alpha, 2);
                              );
    aiatensor__(free)(diff);
    aiatensor__(free)(y);
  }
  aiatensor__(resize2d)(K_, n, m);
  aiatensor__(freeCopyTo)(K_, K);
  return K;
}

T aiakernel_se__(value)(AIATensor_ *x, AIATensor_ *y, T alpha, AIATensor_ *lambda, int isdiag, const char *uplo) {
  if(y == NULL) y = x;

  aia_argcheck(aiatensor__(isVector)(x), 2, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(y), 3, "function only works for vector inputs");
  aia_argcheck(aiatensor__(isVector)(lambda) || aiatensor__(isSquare)(lambda), 4, "incorrect size of lambda");
  aia_argcheck(x->size[0] == y->size[0], 2, "incosistent tensor size");
  aia_argcheck(x->size[0] == lambda->size[0], 4, "incosistent tensor size for lambda");

  T k;

  AIATensor_ *diff = aiatensor__(empty)();
  aiatensor__(csub)(diff, x, 1, y);

  if(isdiag) {
    k = aiatensor__(xTAdiagIx)(diff, lambda);
  } else {
    k = aiatensor__(xTApdIx)(diff, lambda, uplo);
  }
  k *= -0.5;
  k = exp(k) * pow(alpha, 2);
  aiatensor__(free)(diff);
  return k;
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiakernel/kernel.c"
#include <aiautil/erasure.h>