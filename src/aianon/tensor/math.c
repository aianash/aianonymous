#include <aianon/tensor/math.h>

#ifdef ERASED_TYPE_PRESENT

void aiatensor__(add)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = dtnsr[i] + value;
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = *tnsr_data + value;);
  }
}

void aiatensor__(sub)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(add)(res, tnsr, -value);
}

void aiatensor__(mul)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = dtnsr[i] * value;
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = *tnsr_data * value;);
  }
}

void aiatensor__(div)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = dtnsr[i] / value;
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = *tnsr_data / value;);
  }
}

void aiatensor__(fmod)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = fmod(dtnsr[i], value);
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = fmod(*tnsr_data, value););
  }
}

void aiatensor__(remainder)(AIATensor_ *res, AIATensor_ *tnsr, T value) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = (value == 0) ? NAN : dtnsr[i] - value * floor(dtnsr[i] / value);
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = (value == 0) ? NAN :
      *tnsr_data - value * (*tnsr_data / value););
  }
}

void aiatensor__(clamp)(AIATensor_ *res, AIATensor_ *tnsr, T minValue, T maxValue) {
  aiatensor__(resizeAs)(res, tnsr);
  if (aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr) &&
    aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr = aiatensor__(data)(tnsr);
    long sz = aiatensor__(nElement)(tnsr);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < sz; i++) {
      dres[i] = (dtnsr[i] < minValue) ? minValue : (dtnsr[i] > maxValue ? maxValue : dtnsr[i]);
    }
  } else {
    AIA_TENSOR_APPLY2(T, res, T, tnsr, *res_data = (*tnsr_data < minValue) ? minValue :
      (*tnsr_data > maxValue ? maxValue : *tnsr_data););
  }
}

/** res := tnsr1 + alpha * tnsr2 */
void aiatensor__(cadd)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2) {
  aiatensor__(resizeAs)(res, tnsr1);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr1) && aiatensor__(isContiguous)(tnsr2) && aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr2)) {
    if(res == tnsr1) {
      aiablas__(axpy)(aiatensor__(nElement)(tnsr1), alpha, aiatensor__(data)(tnsr2), 1, aiatensor__(data)(res), 1);
    } else {
      T *dres = aiatensor__(data)(res);
      T *dtnsr1 = aiatensor__(data)(tnsr1);
      T *dtnsr2 = aiatensor__(data)(tnsr2);
      long sz = aiatensor__(nElement)(tnsr1);
      long i;
      #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
      for(i = 0; i < sz; i++) {
        dres[i] = alpha * dtnsr2[i] + dtnsr1[1];
      }
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, tnsr1, T, tnsr2, *res_data = *tnsr1_data + alpha * *tnsr2_data;);
  }
}

/** res := tnsr1 - alpha * tnsr2 */
void aiatensor__(csub)(AIATensor_ *res, AIATensor_ *tnsr1, T value, AIATensor_ *tnsr2) {
  aiatensor__(cadd)(res, tnsr1, - value, tnsr2);
}

/** res := tnsr1 * tnsr2 (elementwise multiplication) */
void aiatensor__(cmul)(AIATensor_ *res, AIATensor_ *tnsr1, AIATensor_ *tnsr2) {
  aiatensor__(resizeAs)(res, tnsr1);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr1) && aiatensor__(isContiguous)(tnsr2) && aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr2)) {
    T *dres = aiatensor__(data)(res);
    T *dtnsr1 = aiatensor__(data)(tnsr1);
    T *dtnsr2 = aiatensor__(data)(tnsr2);
    long sz = aiatensor__(nElement)(tnsr1);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = dtnsr1[i] * dtnsr2[i];
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, tnsr1, T, tnsr2, *res_data = *tnsr1_data * *tnsr2_data;);
  }
}

/** res := tnsr1 ^ tnsr2 (elementwise power) */
void aiatensor__(cpow)(AIATensor_ *res, AIATensor_ *base, AIATensor_ *exp) {
  aiatensor__(resizeAs)(res, base);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(base) && aiatensor__(isContiguous)(exp) && aiatensor__(nElement)(res) == aiatensor__(nElement)(exp)) {
    T *dres = aiatensor__(data)(res);
    T *dbase = aiatensor__(data)(base);
    T *dexp = aiatensor__(data)(exp);
    long sz = aiatensor__(nElement)(base);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = pow(dbase[i], dexp[i]);
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, base, T, exp, *res_data = pow(*base_data, *exp_data););
  }
}

/** res := tnsr1 / tnsr2 (elementwise division) */
void aiatensor__(cdiv)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom) {
  aiatensor__(resizeAs)(res, numer);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(numer) && aiatensor__(isContiguous)(denom) && aiatensor__(nElement)(res) == aiatensor__(nElement)(denom)) {
    T *dres = aiatensor__(data)(res);
    T *dnumer = aiatensor__(data)(numer);
    T *ddenom = aiatensor__(data)(denom);
    long sz = aiatensor__(nElement)(numer);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = dnumer[i] / ddenom[i];
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = *numer_data / *denom_data;);
  }
}

/** elementwise modulo */
void aiatensor__(cfmod)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom) {
  aiatensor__(resizeAs)(res, numer);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(numer) && aiatensor__(isContiguous)(denom) && aiatensor__(nElement)(res) == aiatensor__(nElement)(denom)) {
    T *dres = aiatensor__(data)(res);
    T *dnumer = aiatensor__(data)(numer);
    T *ddenom = aiatensor__(data)(denom);
    long sz = aiatensor__(nElement)(numer);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = fmod(dnumer[i], ddenom[i]);
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = fmod(*numer_data, *denom_data););
  }
}

/** elementwise remainder */
void aiatensor__(cremainder)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom) {
  aiatensor__(resizeAs)(res, numer);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(numer) && aiatensor__(isContiguous)(denom) && aiatensor__(nElement)(res) == aiatensor__(nElement)(denom)) {
    T *dres = aiatensor__(data)(res);
    T *dnumer = aiatensor__(data)(numer);
    T *ddenom = aiatensor__(data)(denom);
    long sz = aiatensor__(nElement)(numer);
    long i;
    #pragma omp parallel for if(sz > AIA_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = (ddenom[i] == 0) ? NAN : dnumer[i] - (ddenom[i] * floor(dnumer[i] / ddenom[i]));
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = (*denom_data == 0) ? NAN : *numer_data - (*denom_data * floor(*numer_data / *denom_data)););
  }
}

void aiatensor__(addcmul)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2, AIATensor_ *tnsr3)
{
  if(res != tnsr1)
  {
    aiatensor__(resizeAs)(res, tnsr1);
    aiatensor__(copy)(res, tnsr1);
  }
  AIA_TENSOR_APPLY3(T, res, T, tnsr2, T, tnsr3, *res_data += alpha * *tnsr2_data * *tnsr3_data;);
}


void aiatensor__(addcdiv)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2, AIATensor_ *tnsr3)
{
  if(res != tnsr1)
  {
    aiatensor__(resizeAs)(res, tnsr1);
    aiatensor__(copy)(res, tnsr1);
  }
  AIA_TENSOR_APPLY3(T, res, T, tnsr2, T, tnsr3, *res_data += alpha * *tnsr2_data / *tnsr3_data;);
}

/** res = (beta * bvec) + (alpha * (mat * vec)) */
void aiatensor__(addmv)(AIATensor_ *res, T beta, AIATensor_ *bvec, T alpha, AIATensor_ *mat, AIATensor_ *vec) {
  aia_argcheck(mat->nDimension != 2, 5, "matrix expected got %dD", mat->nDimension);
  aia_argcheck(vec->nDimension != 1, 6, "vector expected got %dD", vec->nDimension);
  aia_argcheck(bvec->nDimension != 1, 3, "vector expected got %dD", bvec->nDimension);
  aia_argcheck(mat->size[1] != vec->size[0], 5, "size mismatch between matrix and vector");
  aia_argcheck(mat->size[0] != bvec->size[0], 5, "size mismatch between matrix and bvector");

  if(res != bvec) {
    aiatensor__(resizeAs)(res, bvec);
    aiatensor__(copy)(res, bvec);
  }

  if(mat->stride[0] == 1) {
    aiablas__(gemv)('n', mat->size[0], mat->size[1],
                    alpha, aiatensor__(data)(mat), mat->stride[1], aiatensor__(data)(vec), vec->stride[0],
                    beta, aiatensor__(data)(res), res->stride[0]);
  } else if(mat->stride[1] == 1) {
    aiablas__(gemv)('t', mat->size[1], mat->size[0],
                    alpha, aiatensor__(data)(mat), mat->stride[0], aiatensor__(data)(vec), vec->stride[0],
                    beta, aiatensor__(data)(res), res->stride[0]);
  } else {
    AIATensor_ *cmat = aiatensor__(contiguous)(mat);
    aiablas__(gemv)('t', mat->size[1], mat->size[0],
                    alpha, aiatensor__(data)(mat), mat->stride[0], aiatensor__(data)(vec), vec->stride[0],
                    beta, aiatensor__(data)(res), res->stride[0]);
    aiatensor__(free)(cmat);
  }
}

/** res = (beta * bmat) + (alpha * mat1 * mat2) */
void aiatensor__(addmm)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *mat1, AIATensor_ *mat2) {
  if(res != bmat) {
    aiatensor__(resizeAs)(res, bmat);
    aiatensor__(copy)(res, bmat);
  }

  char trans_res, trans_mat1, trans_mat2;
  AIATensor_ *res_, *mat1_, *mat2_;

  if(res->stride[0] == 1) {
    trans_res = 'n';
    res_ = res;
  } else if(res->stride[1] == 1) {
    trans_res = 't';
    res_ = res;
    SWAP(AIATensor_ *, mat1, mat2);
  } else {
    trans_res = 'n';
    AIATensor_ *resT = aiatensor__(new)(res);
    aiatensor__(transpose)(resT, NULL, 0, 1);
    res_ = aiatensor__(clone)(resT);
    aiatensor__(free)(resT);
    aiatensor__(transpose)(res_, NULL, 0, 1);
  }

  if(mat1->stride[(trans_res == 'n' ? 0 : 1)] == 1) {
    trans_mat1 = 'n';
    mat1_ = mat1;
  } else if(mat1->stride[(trans_res == 'n' ? 1 : 0)] == 1) {
    trans_mat1 = 't';
    mat1_ = mat1;
  } else {
    trans_mat1 = (trans_res == 'n' ? 't' : 'n');
    mat1_ = aiatensor__(contiguous)(mat1);
  }

  if(mat2->stride[(trans_res == 'n' ? 0 : 1)] == 1) {
    trans_mat2 = 'n';
    mat2_ = mat2;
  } else if(mat2->stride[(trans_res == 'n' ? 1 : 0)] == 1) {
    trans_mat2 = 't';
    mat2_ = mat2;
  } else {
    trans_mat2 = (trans_res == 'n' ? 't' : 'n');
    mat2_ = aiatensor__(contiguous)(mat2);
  }

  aiablas__(gemm)(trans_mat1, trans_mat2,
                  res_->size[(trans_res == 'n' ? 0 : 1)],
                  res_->size[(trans_res == 'n' ? 1 : 0)],
                  mat1_->size[(trans_res == 'n' ? 1 : 0)],
                  alpha,
                  aiatensor__(data)(mat1_),
                  (trans_mat1 == 'n' ? mat1_->stride[(trans_res == 'n' ? 1 : 0)] : mat1_->stride[(trans_res == 'n' ? 0 : 1)]),
                  aiatensor__(data)(mat2_),
                  (trans_mat2 == 'n' ? mat2_->stride[(trans_res == 'n' ? 1 : 0)] : mat2_->stride[(trans_res == 'n' ? 0 : 1)]),
                  beta,
                  aiatensor__(data)(res_),
                  res_->stride[(trans_res == 'n' ? 1 : 0)]);

  if(mat1_ != mat1) aiatensor__(free)(mat1_);
  if(mat2_ != mat2) aiatensor__(free)(mat2_);
  if(res_ != res) aiatensor__(freeCopyTo)(res_, res);
}

/** res = (beta * bmat) + (alpha * vec1 x vec2) */
void aiatensor__(addr)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *vec1, AIATensor_ *vec2) {
  if(res != bmat) {
    aiatensor__(resizeAs)(res, bmat);
    aiatensor__(copy)(res, bmat);
  }

  if(beta != 1) aiatensor__(mul)(res, res, beta);

  if(res->stride[0] == 1) {
    aiablas__(ger)(vec1->size[0], vec2->size[0],
                   alpha, aiatensor__(data)(vec1), vec1->stride[0],
                   aiatensor__(data)(vec2), vec2->stride[0],
                   aiatensor__(data)(res), res->stride[1]);
  } else if(res->stride[1] == 1) {
    aiablas__(ger)(vec2->size[0], vec1->size[0],
                   alpha, aiatensor__(data)(vec2), vec2->stride[0],
                   aiatensor__(data)(vec1), vec1->stride[0],
                   aiatensor__(data)(res), res->stride[0]);

  } else {
    AIATensor_ *cr = aiatensor__(clone)(res);
    aiablas__(ger)(vec2->size[0], vec1->size[0],
                   alpha, aiatensor__(data)(vec2), vec2->stride[0],
                   aiatensor__(data)(vec1), vec1->stride[0],
                   aiatensor__(data)(cr), cr->stride[0]);
    aiatensor__(freeCopyTo)(cr, res);
  }
}

void aiatensor__(addbmm)(AIATensor_ *res, T beta, AIATensor_ *bmat, T alpha, AIATensor_ *batch1, AIATensor_ *batch2) {
  long batchidx;

  aia_argcheck(aiatensor__(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  aia_argcheck(aiatensor__(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  aia_argcheck(aiatensor__(size)(batch1, 0) == aiatensor__(size)(batch2, 0), 2,
    "equal number of batches expected, got %d and %d",
    aiatensor__(size)(batch1, 0), aiatensor__(size)(batch2, 0));
  aia_argcheck(aiatensor__(size)(batch1, 2) == aiatensor__(size)(batch2, 1), 2,
    "wrong matrix size, batch1 = %dx%d and batch2 = %dx%d",
    aiatensor__(size)(batch1, 1), aiatensor__(size)(batch1, 2),
    aiatensor__(size)(batch1, 2), aiatensor__(size)(batch2, 2));

  long dim1 = aiatensor__(size)(batch1, 1);
  long dim2 = aiatensor__(size)(batch2, 2);
  aia_argcheck(aiatensor__(size)(bmat, 0) == dim1, 1, "output tensor of incorrect size");
  aia_argcheck(aiatensor__(size)(bmat, 1) == dim2, 1, "output tensor of incorrect size");

  if(bmat != res) {
    aiatensor__(resizeAs)(res, bmat);
    aiatensor__(copy)(res, bmat);
  }

  AIATensor_ *mat1 = aiatensor__(empty)();
  AIATensor_ *mat2 = aiatensor__(empty)();

  for(batchidx = 0; batchidx < aiatensor__(size)(batch1, 0); ++batchidx) {
    aiatensor__(select)(mat1, batch1, 0, batchidx);
    aiatensor__(select)(mat2, batch2, 0, batchidx);

    aiatensor__(addmm)(res, beta, res, alpha, mat1, mat2);
    beta = 1;
  }

  aiatensor__(free)(mat1);
  aiatensor__(free)(mat2);
}


void aiatensor__(baddbmm)(AIATensor_ *res, T beta, AIATensor_ *batch3, T alpha, AIATensor_ *batch1, AIATensor_ *batch2) {
  long batchidx;

  aia_argcheck(aiatensor__(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  aia_argcheck(aiatensor__(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  aia_argcheck(aiatensor__(size)(batch1, 0) == aiatensor__(size)(batch2, 0), 2,
    "equal number of batches expected, got %d and %d",
    aiatensor__(size)(batch1, 0), aiatensor__(size)(batch2, 0));
  aia_argcheck(aiatensor__(size)(batch1, 2) == aiatensor__(size)(batch2, 1), 2,
    "wrong matrix size, batch1 = %dx%d and batch2 = %dx%d",
    aiatensor__(size)(batch1, 1), aiatensor__(size)(batch1, 2),
    aiatensor__(size)(batch1, 2), aiatensor__(size)(batch2, 2));

  long dim1 = aiatensor__(size)(batch1, 0);
  long dim2 = aiatensor__(size)(batch1, 1);
  long dim3 = aiatensor__(size)(batch2, 2);
  aia_argcheck(aiatensor__(size)(batch3, 0) == dim1, 1, "output tensor size incorrect");
  aia_argcheck(aiatensor__(size)(batch3, 1) == dim2, 1, "output tensor size incorrect");
  aia_argcheck(aiatensor__(size)(batch3, 2) == dim3, 1, "output tensor size incorrect");

  if(res != batch3) {
    aiatensor__(resizeAs)(res, batch3);
    aiatensor__(copy)(res, batch3);
  }

  AIATensor_ *mat1 = aiatensor__(empty)();
  AIATensor_ *mat2 = aiatensor__(empty)();
  AIATensor_ *res_mat = aiatensor__(empty)();

  for(batchidx = 0; batchidx < aiatensor__(size)(batch1, 0); ++batchidx) {
    aiatensor__(select)(mat1, batch1, 0, batchidx);
    aiatensor__(select)(mat2, batch2, 0, batchidx);
    aiatensor__(select)(res_mat, res, 0, batchidx);

    aiatensor__(addmm)(res_mat, beta, res_mat, alpha, mat1, mat2);
  }

  aiatensor__(free)(mat1);
  aiatensor__(free)(mat2);
  aiatensor__(free)(res_mat);
}

void aiatensor__(mv)(AIATensor_ *res, AIATensor_ *mat, AIATensor_ *vec) {}

T aiatensor__(dot)(AIATensor_ *vec1, AIATensor_ *vec2) {
  return 0;
}

//
int aiatensor__(eq)(AIATensor_ *a, AIATensor_ *b) {
  int equal = 1;
  if(!aiatensor__(isSameSizeAs)(a, b)) return 0;
  if(aiatensor__(isContiguous)(a) && aiatensor__(isContiguous)(b)) {
    T *ad = aiatensor__(data)(a);
    T *bd = aiatensor__(data)(b);
    long n = aiatensor__(nElement)(a);
    long i;
    for(i = 0; i < n; i++) {
      if(ad[i] != bd[i]) return 0;
    }
  } else {
    AIA_TENSOR_APPLY2(T, a, T, b,
                      if(equal && *a_data != *b_data) {
                        equal = 0;
                        tensor_apply_finished = 1; break;
                      })
  }
  return equal;
}

#if defined(T_IS_FLOAT) || defined(T_IS_DOUBLE)
//
int aiatensor__(epsieq)(AIATensor_ *a, AIATensor_ *b, T epsi) {
  int equal = 1;
  if(!aiatensor__(isSameSizeAs)(a, b)) return 0;
  if(epsi < 0) epsi = -epsi;
  if(aiatensor__(isContiguous)(a) && aiatensor__(isContiguous)(b)) {
    T *ad = aiatensor__(data)(a);
    T *bd = aiatensor__(data)(b);
    long n = aiatensor__(nElement)(a);
    long i;
    for(i = 0; i < n; i++) {
#ifdef T_IS_FLOAT
      if(fabsf(ad[i] - bd[i]) > epsi) return 0;
#elif defined(T_IS_DOUBLE)
      if(fabs(ad[i] - bd[i]) > epsi) return 0;
#endif
    }
  } else {
#ifdef T_IS_FLOAT
    AIA_TENSOR_APPLY2(T, a, T, b,
                      if(equal && fabsf(a_data - b_data) > epsi) {
                        equal = 0;
                        tensor_apply_finished = 1; break;
                      })
#elif defined(T_IS_DOUBLE)
    AIA_TENSOR_APPLY2(T, a, T, b,
                      if(equal && fabs(a_data - b_data) > epsi) {
                        equal = 0;
                        tensor_apply_finished = 1; break;
                      })
#endif
  }
  return equal;
}
#endif

T aiatensor__(trace)(AIATensor_ *mat) {
  aia_argcheck(aiatensor__(isSquare)(mat), 1, "A should be 2-dimensional");

  T *mat_data = aiatensor__(data)(mat);
  long stride = mat->stride[0] + mat->stride[1];
  long idx;
  T tr = 0;

  for(idx = 0; idx < mat->size[0]; idx++) {
    tr += mat_data[idx * stride];
  }

  return tr;
}

//
void aiatensor__(fill)(AIATensor_ *res, T value) {
  AIA_TENSOR_APPLY(T, res,
                  {
                    arr__(fill)(res_data, value, res_size); break;
                  })
}

//
void aiatensor__(zero)(AIATensor_ *res) {
  AIA_TENSOR_APPLY(T, res,
                  {
                    arr__(zero)(res_data, res_size); break;
                  })
}

//
void aiatensor__(maskedFill)(AIATensor_ *res, AIATensor(uchar) *mask, T value) {
  AIA_TENSOR_APPLY2(T, res, unsigned char, mask, if(*mask_data == 1) *res_data = value;)
}

//
void aiatensor__(maskedCopy)(AIATensor_ *res, AIATensor(uchar) *mask, AIATensor_ *from) {
  long nele_res = aiatensor__(nElement)(res);
  long nele_mask = aiatensor_(uchar, nElement)(mask);
  long nele_from = aiatensor__(nElement)(from);
  aia_argcheck(nele_res == nele_mask && nele_res == nele_from, 1,
    "Number of elements in res, mask and from should be equal, use narrow");

  AIA_TENSOR_APPLY3(T, res, unsigned char, mask , T, from,
    {
      if(*mask_data == 1)
        *res_data = *from_data;
    }
  )
}

//
void aiatensor__(zeros)(AIATensor_ *res, int nDimension, long *size, long *stride) {
  aiatensor__(resize)(res, nDimension, size, stride);
  aiatensor__(zero)(res);
}

//
void aiatensor__(ones)(AIATensor_ *res, int nDimension, long *size, long *stride) {
  aiatensor__(resize)(res, nDimension, size, stride);
  aiatensor__(fill)(res, (T)1.0);
}

/** det(mat) */
T aiatensor__(detsymm)(AIATensor_ *mat) {
  aia_argcheck(aiatensor__(isSquare)(mat), 1, "A should be square matrix");

  AIATensor_ *matchol = aiatensor__(new)(mat);
  T *data = aiatensor__(data)(matchol);
  long stride = matchol->stride[0] + matchol->stride[1];
  T det = 1;
  long idx;

  aiatensor__(potrf)(matchol, matchol, "L");
  for(idx = 0; idx < matchol->size[0]; idx++) {
    det *= data[idx * stride];
  }
  det = pow(det, 2);
  aiatensor__(free)(matchol);
  return det;
}

/** X + a * I */
void aiatensor__(aIpX)(AIATensor_ *res, AIATensor_ *mat, T a) {
  if(mat == NULL) mat = res;
  aia_argcheck(aiatensor__(isMatrix)(mat), 2, "mat should be 2-dimensional");
  aia_argcheck(mat->size[0] == mat->size[1], 2, "mat should be a square matrix");

  if(mat != res) {
    aiatensor__(resizeAs)(res, mat);
    aiatensor__(copy)(res, mat);
  }

  T *res_data = aiatensor__(data)(res);
  long idx;
  for(idx = 0; idx < res->size[0]; idx++) {
    *res_data += a;
    res_data += (res->stride[0] + res->stride[1]);
  }
}

/** x** * A * x */
T aiatensor__(xTAx)(AIATensor_ *x, AIATensor_ *amat) {
  return aiatensor__(xTAy)(x, amat, x);
}

/** x** * A * y */
T aiatensor__(xTAy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y) {
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be 1-dimensional");
  aia_argcheck(aiatensor__(isMatrix)(amat), 2, "A should be a matrix");
  aia_argcheck(aiatensor__(isVector)(y), 3, "y should be 1-dimensional");
  aia_argcheck(x->size[0] == amat->size[0], 2, "inconsistent tensor size");
  aia_argcheck(y->size[0] == amat->size[1], 2, "inconsistent tensor size");

  long nrows = amat->size[0];
  long lda = nrows;
  AIATensor_ *z = aiatensor__(newVector)(nrows);
  T res;

  aiatensor__(mv)(z, amat, y);
  res = aiatensor__(dot)(z, x);
  aiatensor__(free)(z);
  return res;
}

/** x** * A^-1 * x */
T aiatensor__(xTAIx)(AIATensor_ *x, AIATensor_ *amat) {
  return aiatensor__(xTAIy)(x, amat, x);
}

/** x** * A^-1 * y */
T aiatensor__(xTAIy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y) {
  printf("ERR: function aiatensor__(xTAIy) is not implemented");
  exit(-1);
}

/** x** * Asymm * x */
T aiatensor__(xTAsymmx)(AIATensor_ *x, AIATensor_ *amat) {
  printf("ERR: function aiatensor__(xTAsymmx) is not implemented");
  exit(-1);
}

/** x** * Asymm * y */
T aiatensor__(xTAsymmy)(AIATensor_ *x, AIATensor_ *amat, AIATensor_ *y) {
  printf("ERR: function aiatensor__(xTAsymmy) is not implemented");
  exit(-1);
}

/** x** * Apd^-1 * x */
T aiatensor__(xTApdIx)(AIATensor_ *x, AIATensor_ *achol, const char *uplo) {
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be 1-dimensional");
  aia_argcheck(aiatensor__(isSquare)(achol), 2, "A should be square matrix");
  aia_argcheck(x->size[0] == achol->size[0], 2, "inconsistent tensor size");

  AIATensor_ *L = aiatensor__(new)(achol);
  AIATensor_ *LIx = aiatensor__(newVector)(achol->size[0]);
  T res;

  aiatensor__(potrf)(L, achol, "L");
  aiatensor__(trtrs)(LIx, x, L, "L", "N", "N");
  res = aiatensor__(dot)(LIx, LIx);
  aiatensor__(free)(L);
  aiatensor__(free)(LIx);
  return res;
}

/** x** * Apd^-1 * y */
T aiatensor__(xTAsymmIy)(AIATensor_ *x, AIATensor_ *achol, const char *uplo, AIATensor_ *y) {
  aia_argcheck(aiatensor__(isVector)(x), 1, "x should be 1-dimensional");
  aia_argcheck(aiatensor__(isSquare)(achol), 2, "A should be square matrix");
  aia_argcheck(aiatensor__(isVector)(y), 3, "y should be 1-dimensional");
  aia_argcheck(x->size[0] == achol->size[0], 2, "inconsistent tensor size");
  aia_argcheck(y->size[0] == achol->size[1], 2, "inconsistent tensor size");

  AIATensor_ *L = aiatensor__(new)(achol);
  AIATensor_ *LIy = aiatensor__(newVector)(achol->size[0]);
  T res;

  aiatensor__(potrf)(L, achol, "L");
  aiatensor__(potrs)(LIy, y, L, "L");
  res = aiatensor__(dot)(LIy, x);
  return res;
}

/** X** * Asymm * X + a * Y */
AIATensor_ *aiatensor__(XTAsymmXpaY)(AIATensor_ *res, AIATensor_ *xmat, AIATensor_ *amat, T a, AIATensor_ *ymat) {
  printf("ERR: function aiatensor__(XTAsymmXpaY) is not implemented");
  exit(-1);
  return res;
}

/** X** * Apd^-1 * X + a * Y */
AIATensor_ *aiatensor__(XTApdIXpaY)(AIATensor_ *res, AIATensor_ *xmat, AIATensor_ *achol, const char *uplo, T a, AIATensor_ *ymat) {
  aia_argcheck(aiatensor__(isSquare)(xmat), 2, "X should be square matrix");
  aia_argcheck(aiatensor__(isSquare)(achol), 3, "A should be square matrix");
  aia_argcheck(aiatensor__(isSquare)(ymat), 6, "Y should be square matrix");
  aia_argcheck(xmat->size[0] == achol->size[0], 2, "inconsistent tensor size");
  aia_argcheck(xmat->size[0] == ymat->size[0], 3, "inconsistent tensor size");

  AIATensor_ *aIx  = aiatensor__(empty)();
  AIATensor_ *aIxT = aiatensor__(empty)();

  aiatensor__(trtrs)(aIx, xmat, achol, uplo, "N", "N");
  aiatensor__(transpose)(aIxT, aIx, 0, 1);
  aiatensor__(addmm)(res, a, ymat, 1, aIx, aIxT);

  aiatensor__(free)(aIx);
  aiatensor__(free)(aIxT);
  return res;
}

#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor/math.c"
#include <aianon/core/erasure.h>