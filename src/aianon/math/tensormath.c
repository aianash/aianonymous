#include <aianon/math/tensormath.h>
#include <aianon/math/blas.h>
#include <math.h>

#ifdef ERASED_TYPE_AVAILABLE

#define TH_OMP_OVERHEAD_THRESHOLD 100000
#define AIA_TENSOR_APPLY3(...) {}

void aiatensor__(resizeAs)(AIATensor_ *res, AIATensor_ *source) {

}

void aiatensor__(copy)(AIATensor_ *res, AIATensor_ *source) {

}

void aiatensor__(select)(AIATensor_ *res, AIATensor_ *src, long a, long b) {

}

int aiatensor__(isContiguous)(AIATensor_ *ten) {
  return 1;
}

long aiatensor__(nDimension)(AIATensor_ *ten) {
  return 1;
}

long aiatensor__(size)(AIATensor_ *t, long dim) {
  return 1;
}

T* aiatensor__(data)(AIATensor_ *t) {
  return (T*) 0;
}

long aiatensor__(nElement)(AIATensor_ *t) {
  return 2;
}

/** res := tnsr1 + alpha * tnsr2 */
void aiatensor__(cadd)(AIATensor_ *res, AIATensor_ *tnsr1, T alpha, AIATensor_ *tnsr2) {
  aiatensor__(resizeAs)(res, tnsr1);
  if(aiatensor__(isContiguous)(res) && aiatensor__(isContiguous)(tnsr1) && aiatensor__(isContiguous)(tnsr2) && aiatensor__(nElement)(res) == aiatensor__(nElement)(tnsr2)) {
    if(res == tnsr1) {
      aiablas_(T_, axpy)(aiatensor__(nElement)(tnsr1), alpha, aiatensor__(data)(tnsr2), 1, aiatensor__(data)(res), 1);
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


void aiatensor__(addbmm)(AIATensor_ *res, T beta, AIATensor_ *mat, T alpha, AIATensor_ *batch1, AIATensor_ *batch2) {
  long batchidx;

  aiaargcheck(aiatensor__(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  aiaargcheck(aiatensor__(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  aiaargcheck(aiatensor__(nSize)(batch1, 0) == aiatensor__(nSize)(batch2, 0), 2,
    "equal number of batches expected, got %d and %d",
    aiatensor__(nSize)(batch1, 0), aiatensor__(nSize)(batch2, 0));
  aiaargcheck(aiatensor__(nSize)(batch1, 2) == aiatensor__(nSize)(batch2, 1), 2,
    "wrong matrix size, batch1 = %dx%d and batch2 = %dx%d",
    aiatensor__(nSize)(batch1, 1), aiatensor__(nSize)(batch1, 2),
    aiatensor__(nSize)(batch1, 2), aiatensor__(nSize)(batch2, 2));

  long dim1 = aiatensor__(nSize)(batch1, 1);
  long dim2 = aiatensor__(nSize)(batch2, 2);
  aiaargcheck(aiatensor__(nSize)(mat, 0) == dim1, "output tensor of incorrect size");
  aiaargcheck(aiatensor__(nSize)(mat, 1) == dim2, "output tensor of incorrect size");

  if(mat != res) {
    aiatensor__(resizeAs)(res, mat);
    aiatensor__(copy)(res, mat);
  }

  AIATensor_ *mat1 = aiatensor__(empty)();
  AIATensor_ *mat2 = aiatensor__(empty)();

  for(batchidx = 0; batchidx < aiatensor__(nSize)(batch1, 0); ++batchidx) {
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

  aiaargcheck(aiatensor__(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  aiaargcheck(aiatensor__(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  aiaargcheck(aiatensor__(nSize)(batch1, 0) == aiatensor__(nSize)(batch2, 0), 2,
    "equal number of batches expected, got %d and %d",
    aiatensor__(nSize)(batch1, 0), aiatensor__(nSize)(batch2, 0));
  aiaargcheck(aiatensor__(nSize)(batch1, 2) == aiatensor__(nSize)(batch2, 1), 2,
    "wrong matrix size, batch1 = %dx%d and batch2 = %dx%d",
    aiatensor__(nSize)(batch1, 1), aiatensor__(nSize)(batch1, 2),
    aiatensor__(nSize)(batch1, 2), aiatensor__(nSize)(batch2, 2));

  long dim1 = aiatensor__(nSize)(batch1, 0);
  long dim2 = aiatensor__(nSize)(batch1, 1);
  long dim3 = aiatensor__(nSize)(batch2, 2);
  aiaargcheck(aiatensor__(nSize)(batch3, 0) == dim1, "output tensor size incorrect");
  aiaargcheck(aiatensor__(nSize)(batch3, 1) == dim2, "output tensor size incorrect");
  aiaargcheck(aiatensor__(nSize)(batch3, 2) == dim3, "output tensor size incorrect");

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
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/math/tensormath.c"
#include <aianon/util/erasure.h>