#include <aianon/tensor/math.h>

#ifdef ERASED_TYPE_PRESENT

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

#endif
#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/math.c"
#include <aianon/core/erasure.h>