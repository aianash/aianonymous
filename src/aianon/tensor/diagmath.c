#include <aianon/tensor/diagmath.h>

#ifdef ERASED_TYPE_PRESENT

void aiatensor__(diagmm)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *dmat, bool isinv) {
  aia_argcheck(aiatensor__(isMatrix)(mat), 2, "mat should be 2-dimensional");
  aia_argcheck(dmat->nDimension == 1, 3, "empty diagonal matrix");
  aia_argcheck(mat->size[1] == dmat->size[0], 3, "inconsistent tensor size");

  aiatensor__(resizeAs)(res, mat);

  T *dmat_data = aiatensor__(data)(dmat);
  long dmat_stride = dmat->stride[0];
  int exp = isinv ? -1 : 1;

  AIA_TENSOR_DIM_APPLY2(T, mat, T, res, 0,
      aiablas__(copy)(mat->size[0], mat_data, mat_stride, res_data, res_stride);
      aiablas__(scal)(res->size[0], pow(*dmat_data, exp), res_data, res_stride);
      dmat_data += dmat_stride;
    );
}

void aiatensor__(cadddiag)(AIATensor_ *res, AIATensor_ *mat, T alpha, AIATensor_ *dmat) {
  aia_argcheck(aiatensor__(isSquare)(mat), 2, "mat should be a square matrix");
  aia_argcheck(dmat->nDimension == 1, 3, "incorrect diagonal matrix");
  aia_argcheck(mat->size[1] == dmat->size[0], 3, "inconsistent tensor size");

  aiatensor__(resizeAs)(res, mat);
  aiatensor__(copy)(res, mat);

  T *dmat_data = aiatensor__(data)(dmat);
  T *res_data = aiatensor__(data)(res);
  long total_stride = res->stride[0] + res->stride[1];
  long idx;

  for(idx = 0; idx < res->size[0]; idx++) {
    res_data[idx * total_stride] += alpha * dmat_data[idx * dmat->stride[0]];
  }
}

void aiatensor__(diaginv)(AIATensor_ *matinv, AIATensor_ *mat) {
  aia_argcheck(mat->nDimension == 1, 3, "incorrect diagonal matrix");

  aiatensor__(resizeAs)(matinv, mat);
  AIA_TENSOR_APPLY2(T, mat, T, matinv, *matinv_data = pow(*mat_data, -1););
}

T aiatensor__(xTAdiagx)(AIATensor_ *x, AIATensor_ *dmat) {
  return aiatensor__(xTAdiagy)(x, dmat, x);
}

T aiatensor__(xTAdiagy)(AIATensor_ *x, AIATensor_ *dmat, AIATensor_ *y) {
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be a vector");
  aia_argcheck(aiatensor__(isVector)(y), 3, "y should be a vector");
  aia_argcheck(dmat->nDimension == 1, 3, "incorrect diagonal matrix");
  aia_argcheck(x->size[0] == dmat->size[0], 1, "inconsistent tensor size");
  aia_argcheck(y->size[0] == dmat->size[0], 3, "inconsistent tensor size");

  T sum = 0;
  AIA_TENSOR_APPLY3(T, x, T, dmat, T, y, sum += (*x_data * *dmat_data * *y_data););
  return sum;
}

T aiatensor__(xTAdiagIx)(AIATensor_ *x, AIATensor_ *dmat) {
  return aiatensor__(xTAdiagIy)(x, dmat, x);
}

T aiatensor__(xTAdiagIy)(AIATensor_ *x, AIATensor_ *dmat, AIATensor_ *y) {
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be a vector");
  aia_argcheck(aiatensor__(isVector)(y), 3, "y should be a vector");
  aia_argcheck(dmat->nDimension == 1, 3, "incorrect diagonal matrix");
  aia_argcheck(x->size[0] == dmat->size[0], 1, "inconsistent tensor size");
  aia_argcheck(y->size[0] == dmat->size[0], 3, "inconsistent tensor size");

  T sum = 0;
  AIA_TENSOR_APPLY3(T, x, T, dmat, T, y, sum += (*x_data * pow(*dmat_data, -1) * *y_data););
  return sum;
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/diagmath.c"
#include <aianon/core/erasure.h>