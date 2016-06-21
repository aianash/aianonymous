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

  work = aiatensor__(newVector)(lwork);
  aialapack__(syev)(jobz[0], uplo[0], n, aiatensor__(data)(resv_), lda,
                    aiatensor__(data)(rese), aiatensor__(data)(work), lwork, &info);

  aia_lapackCheckWithCleanup("Lapack Error %s : %d off-diagonal elements didn't converge to zero",
                             aia_cleanup(aiatensor__(free)(resv_);
                                         aiatensor__(free)(work);),
                             "syev", info, "");
  aiatensor__(freeCopyTo)(resv_, resv);
  aiatensor__(free)(work);
}

#endif
#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/linalg.c"
#include <aianon/core/erasure.h>