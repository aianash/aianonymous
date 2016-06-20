#include <aianon/tensor/math.h>

#ifdef ERASED_TYPE_PRESENT


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
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for(i = 0; i < sz; i++) {
        dres[i] = alpha * dtnsr2[i] + dtnsr1[1];
      }
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, tnsr1, T, tnsr2, *res_data = *tnsr1_data + value * *tnsr2_data);
  }
}

/** res := tnsr1 + alpha * tnsr2 */
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
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = dtnsr1[i] * dtnsr2[i];
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, tnsr1, T, tnsr2, *res_data = *tnsr1_data * *tnsr2_data);
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
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = pow(dbase[i], dexp[i]);
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, base, T, exp, *res_data = pow(*base_data, *exp_data));
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
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = dnumer[i] / ddenom[i];
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = *numer_data / *denom_data);
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
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = fmod(dnumer[i], ddenom[i]);
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = fmod(*numer_data, *denom_data));
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
    #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for(i = 0; i < sz; i++) {
      dres[i] = (ddenom[i] == 0) ? NAN : dnumer[i] - (ddenom[i] * floor(dnumer[i] / ddenom[i]));
    }
  } else {
    AIA_TENSOR_APPLY3(T, res, T, numer, T, denom, *res_data = (*denom_data == 0) ? NAN : *numer_data - (*denom_data * floor(*numer_data / *denom_data)));
  }
}

// res = (beta * bvec) + (alpha * (mat * vec))
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

// res = (beta * bmat) + (alpha * mat1 * mat2)
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
    AIATensor_ *resT = aiatensor__(newTransposed)(res, 0, 1);
    res_ = aiatensor__(clone)(resT);
    aiatensor__(free)(resT);
    aiatensor__(transpose)(res_, 0, 1);
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

// res = (beta * bmat) + (alpha * vec1 x vec2)
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


#endif
#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/math.c"
#include <aianon/core/erasure.h>