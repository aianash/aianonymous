#include <aianon/tensor/linalg.h>

#ifdef ERASED_TYPE_PRESENT

// Check if self is transpose of a contiguous matrix
static int aiatensor__(isTransposedContiguous)(AIATensor_ *this) {
  return this->stride[0] == 1 && this->stride[1] == this->size[0];
}

// If a matrix is a regular contiguous matrix, make sure it is transposed
// because this is what we return from Lapack calls.
static void aiatensor__(checkTransposed)(AIATensor_ *this) {
  if(aiatensor__(isContiguous)(this)) aiatensor__(transpose)(this, NULL, 0, 1);
}

// newContiguous followed by transpose
// Similar to (newContiguous), but checks if the transpose of the matrix
// is contiguous and also limited to 2D matrices.
static AIATensor_ *aiatensor__(newTransposedContiguous)(AIATensor_ *this) {
  AIATensor_ *tensor;
  if(aiatensor__(isTransposedContiguous)(this)) {
    aiatensor__(retain)(this);
    tensor = this;
  } else {
    tensor = aiatensor__(contiguous)(this);
    aiatensor__(transpose)(tensor, NULL, 0, 1);
  }

  return tensor;
}

// Given the result tensor and src tensor, decide if the lapack call should use the
// provided result tensor or should allocate a new space to put the result in.
// The returned tensor have to be freed by the calling function.
// nrows is required, because some lapack calls, require output space smaller than
// input space, like underdetermined gels.
static AIATensor_ *aiatensor__(checkLapackClone)(AIATensor_ *result, AIATensor_ *src, int nrows) {
  if(src == result && aiatensor__(isTransposedContiguous)(src) && src->size[1] == nrows)
    aiatensor__(retain)(result);
  else if(src == result || result == NULL)
    result = aiatensor__(empty)();
  else
    aiatensor__(retain)(result);
  return result;
}

// Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
// the resulting tensor to be larger than src.
static AIATensor_ *aiatensor__(cloneColumnMajorNrows)(AIATensor_ *this, AIATensor_ *src, int nrows) {
  AIATensor_ *result;
  AIATensor_ *view;

  if(src == NULL) src = this;
  result = aiatensor__(checkLapackClone)(this, src, nrows);
  if(src == result) return result;

  aiatensor__(resize2d)(result, src->size[1], nrows);
  aiatensor__(checkTransposed)(result);

  if(src->size[0] == nrows) aiatensor__(copy)(result, src);
  else {
    view = aiatensor__(new)(result);
    aiatensor__(narrow)(view, NULL, 0, 0, src->size[0]);
    aiatensor__(copy)(view, src);
    aiatensor__(free)(view);
  }

  return result;
}

// Create a clone of src in self column major order for use with Lapack.
// If src == self, a new tensor is allocated, in any case, the return tensor should be
// freed by calling function.
static AIATensor_ *aiatensor__(cloneColumnMajor)(AIATensor_ *this, AIATensor_ *src) {
  return aiatensor__(cloneColumnMajorNrows)(this, src, src->size[0]);
}

static void aiatensor__(clearUpLoTriangle)(AIATensor_ *mat, const char *uplo) {
  aia_argcheck(mat->nDimension == 2, 1, "Mat should be a 2 dimensional");
  aia_argcheck(mat->size[0] == mat->size[1], 1, "A should be a square");

  int n = mat->size[0];

  T *p = aiatensor__(data)(mat);
  long i, j;

  if(uplo[0] == 'U') {
    for(i = 0; i < n; i++)
      for(j = i + 1; j < n; j++)
        p[n*i + j] = 0;
  } else if(uplo[0] == 'L') {
    for(i = 0; i < n; i++)
      for(j = 0; j < i; j++)
        p[n*i + j] = 0;
  }
}

//
void aiatensor__(potrf)(AIATensor_ *res, AIATensor_ *mat, const char *uplo) {
  if(!mat) mat = res;
  aia_argcheck(res->nDimension == 2, 1, "mat or res should be a matrix");
  aia_argcheck(res->size[0] == res->size[1], 1, "mat or res should be a square matrix");

  int n, lda, info;
  AIATensor_ *res_ = aiatensor__(cloneColumnMajor)(res, mat);

  n = res_->size[0];
  lda = n;

  aialapack__(potrf)(uplo[0], n, aiatensor__(data)(res_), lda, &info);
  aia_lapackCheckWithCleanup("Lapack Error %s : A(%d, %d) is 0, A cannot be factorized",
                             aia_cleanup(aiatensor__(free)(res_);),
                             "potrf", info, info);

  aiatensor__(clearUpLoTriangle)(res_, uplo);
  aiatensor__(freeCopyTo)(res_, res);
}

//
void aiatensor__(potrs)(AIATensor_ *res, AIATensor_ *b, AIATensor_ *achol, const char *uplo) {
  if(b == NULL) b = res;
  aia_argcheck(aiatensor__(isSquare)(achol), 3, "A should be a square matrix");

  int n, nrhs, lda, ldb, info;

  AIATensor_ *a_;
  AIATensor_ *b_;

  a_ = aiatensor__(cloneColumnMajor)(NULL, achol);
  b_ = aiatensor__(cloneColumnMajor)(res, b);

  n = a_->size[0];
  nrhs = b_->size[1];
  lda = n;
  ldb = n;
  aialapack__(potrs)(uplo[0], n, nrhs, aiatensor__(data)(a_), lda, aiatensor__(data)(b_), ldb, &info);
  aia_lapackCheckWithCleanup("Lapack Error in %s : A(%d,%d) is zero, singular A",
                            aia_cleanup(
                              aiatensor__(free)(a_);
                              aiatensor__(free)(b_);
                            ), "potrs", info, info);
  aiatensor__(free)(a_);
  aiatensor__(freeCopyTo)(b_, res);
}

void aiatensor__(trtrs)(AIATensor_ *resa, AIATensor_ *resb, AIATensor_ *b, AIATensor_ *amat, const char *uplo, const char *trans, const char *diag) {
  if(amat == NULL) amat = resa;
  if(b == NULL) b = resb;

  aia_argcheck(aiatensor__(isSquare)(amat), 4, "A should be 2-dimensional");
  aia_argcheck(aiatensor__(isMatrix)(b) || aiatensor__(isVector)(b), 3, "b should be either a matrix or a vector");
  aia_argcheck(amat->size[0] == b->size[0], 3, "A, b size incomatible");

  int n, nrhs, lda, ldb, info;
  AIATensor_ *resa_, *resb_;

  resa_ = aiatensor__(cloneColumnMajor)(resa, amat);
  if(aiatensor__(isVector)(b)) {
    resb_ = aiatensor__(newCopy)(b);
    nrhs = 1;
  } else {
    resb_ = aiatensor__(cloneColumnMajor)(resb, b);
    nrhs = resb_->size[1];
  }

  n = (int) resa_->size[0];
  lda = n;
  ldb = n;

  aialapack__(trtrs)(uplo[0], trans[0], diag[0], n, nrhs, aiatensor__(data)(resa_), lda,
    aiatensor__(data)(resb_), ldb, &info);

  aia_lapackCheckWithCleanup("Lapack Error in %s : A(%d, %d) is zero. singular A.",
                             aia_cleanup(aiatensor__(free)(resa_); aiatensor__(free)(resb_);),
                             "trtrs", info, info);

  aiatensor__(freeCopyTo)(resa_, resa);
  aiatensor__(freeCopyTo)(resb_, resb);
}

//
void aiatensor__(syev)(AIATensor_ *rese, AIATensor_ *resv, AIATensor_ *mat, const char *jobz, const char *uplo) {
  if(!mat) mat = resv;
  aia_argcheck(mat->nDimension == 2, 1, "mat should be 2 dimensional");

  int n, lda, lwork, info;
  AIATensor_ *work;

  AIATensor_ *resv_ = aiatensor__(cloneColumnMajor)(resv, mat);

  n = resv_->size[0];
  lda = n;

  aiatensor__(resize1d)(rese, n);

  T optLwork;
  aialapack__(syev)(jobz[0], uplo[0], n, aiatensor__(data)(resv_), lda,
                    aiatensor__(data)(rese), &optLwork, -1, &info);

  lwork = (int)optLwork;

  work = aiatensor__(emptyVector)(lwork);
  aialapack__(syev)(jobz[0], uplo[0], n, aiatensor__(data)(resv_), lda,
                    aiatensor__(data)(rese), aiatensor__(data)(work), lwork, &info);

  aia_lapackCheckWithCleanup("Lapack Error %s : %d off-diagonal elements didn't converge to zero",
                             aia_cleanup(aiatensor__(free)(resv_);
                                         aiatensor__(free)(work);),
                             "syev", info, "");
  aiatensor__(freeCopyTo)(resv_, resv);
  aiatensor__(free)(work);
}


void aiatensor__(gesvd)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *mat, const char *jobu) {
  AIATensor_ *rmat = aiatensor__(empty)();
  aiatensor__(gesvd2)(resu, ress, resv, rmat, mat, jobu);
  aiatensor__(free)(rmat);
}

void aiatensor__(gesvd2)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *rmat, AIATensor_ *mat, const char *jobu) {
  if(mat == NULL) mat = rmat;
  aia_argcheck(aiatensor__(nDimension)(mat) == 2, 1, "A should be 2 dimensional");

  int k, m, n, lda, ldu, ldvt, lwork, info;
  AIATensor_ *work;
  AIATensor_ *rvf_ = aiatensor__(empty)();
  T wkopt;

  AIATensor_ *rmat_ = NULL;
  AIATensor_ *resu_ = NULL;
  AIATensor_ *ress_ = NULL;
  AIATensor_ *resv_ = NULL;

  rmat_ = aiatensor__(cloneColumnMajor)(rmat, mat);

  m = aiatensor__(size)(rmat_, 0);
  n = aiatensor__(size)(rmat_, 1);
  k = (m < n) ? m : n;

  lda = m;
  ldu = m;
  ldvt = n;

  aiatensor__(resize1d)(ress, k);
  aiatensor__(resize2d)(rvf_, ldvt, n);
  if(*jobu == 'A')
    aiatensor__(resize2d)(resu, m, ldu);
  else
    aiatensor__(resize2d)(resu, k, ldu);

  aiatensor__(checkTransposed)(resu);

  resu_ = aiatensor__(newTransposedContiguous)(resu);
  ress_ = aiatensor__(contiguous)(ress);
  resv_ = aiatensor__(contiguous)(rvf_);

  aialapack__(gesvd)(jobu[0], jobu[0], m, n, aiatensor__(data)(rmat_),
    lda, aiatensor__(data)(ress_), aiatensor__(data)(resu_), ldu,
    aiatensor__(data)(resv_), ldvt, &wkopt, -1, &info);

  lwork = (int) wkopt;
  work = aiatensor__(emptyVector)(lwork);

  aialapack__(gesvd)(jobu[0], jobu[0], m, n, aiatensor__(data)(rmat_),
    lda, aiatensor__(data)(ress_), aiatensor__(data)(resu_), ldu,
    aiatensor__(data)(resv_), ldvt, aiatensor__(data)(work), lwork, &info);

  aia_lapackCheckWithCleanup("Lapack error %s : %d superdiagonals failed to converge",
                              aia_cleanup(
                                aiatensor__(free)(resu_);
                                aiatensor__(free)(ress_);
                                aiatensor__(free)(resv_);
                                aiatensor__(free)(rmat_);
                                aiatensor__(free)(work);
                              ), "gesvd", info, "");

  if(*jobu == 'S') aiatensor__(narrow)(resv_, NULL, 1, 0, k);

  aiatensor__(freeCopyTo)(resu_, resu);
  aiatensor__(freeCopyTo)(ress_, ress);
  aiatensor__(freeCopyTo)(resv_, rvf_);
  aiatensor__(freeCopyTo)(rmat_, rmat);

  if(*jobu == 'S') aiatensor__(narrow)(rvf_, NULL, 1, 0, k);

  aiatensor__(resizeAs)(resv, rvf_);
  aiatensor__(copy)(resv, rvf_);
  aiatensor__(free)(rvf_);
}


#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/tensor/linalg.c"
#include <aianon/core/erasure.h>