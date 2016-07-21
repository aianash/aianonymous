#include <aiarandom/random.h>

#ifdef ERASED_TYPE_PRESENT

void aiatensor__(random)(AIATensor_ *res, AIARandGen *gen) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandgen_random(gen););
  /* TODO: check for trucating */
}

void aiatensor__(geometric)(AIATensor_ *res, AIARandGen *gen, T p) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(geometric)(gen, p););
}

void aiatensor__(bernoulli)(AIATensor_ *res, AIARandGen *gen, T p) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(bernoulli)(gen, p););
}

void aiatensor__(uniform)(AIATensor_ *res, AIARandGen *gen, T a, T b) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(uniform)(gen, a, b););
}

void aiatensor__(normal)(AIATensor_ *res, AIARandGen *gen, T mean, T stdv) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(normal)(gen, mean, stdv););
}

void aiatensor__(exponential)(AIATensor_ *res, AIARandGen *gen, T lambda) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(exponential)(gen, lambda););
}

void aiatensor__(cauchy)(AIATensor_ *res, AIARandGen *gen, T median, T sigma) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(cauchy)(gen, median, sigma););
}

void aiatensor__(logNormal)(AIATensor_ *res, AIARandGen *gen, T mean, T stdv) {
  AIA_TENSOR_APPLY(T, res, *res_data = (T)aiarandom__(logNormal)(gen, mean, stdv););
}

/*  Transform matrix of standard normals into matrix where each row
    contains multivariate normals with the desired covariance.
    Compute A such that dot(transpose(A),A) == cov.
    Then the matrix products of the rows of x and A has the desired
    covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
    decomposition of cov is such an A. */
void aiatensor__(mvnormal)(AIATensor_ *res, AIARandGen *gen, AIATensor_ *mean, AIATensor_ *cov) {
  aia_argcheck(res->nDimension == 2, 1, "result must have 2 dimensions");
  aia_argcheck(res->size[1] == mean->size[0], 1, "result must have second dimension size same as mean size");
  aia_argcheck(mean->nDimension == 1, 3, "mean should have 1 dimension");
  aia_argcheck(cov->nDimension == 2, 4, "cov should have 2 dimensions");
  aia_argcheck(cov->size[0] == cov->size[1], 4, "cov should be a square matrix");
  aia_argcheck(cov->size[0] == mean->size[0], 3, "mean and cov have different sizes");

  /* create a tensor of same dimensions as res */
  AIATensor_ *tmp = aiatensor__(emptyAs)(res);

  /* generate a tensor of independent standard normally distributed random numbers */
  aiatensor__(normal)(tmp, gen, 0, 1);

  /* calculate svd decomposition of covariance matrix */
  AIATensor_ *u = aiatensor__(emptyAs)(cov);
  AIATensor_ *s = aiatensor__(emptyVector)(mean->size[0]);
  AIATensor_ *v = aiatensor__(emptyAs)(cov);
  aiatensor__(gesvd)(u, s, v, cov, "A");
  /* set negative singular values to 0 */
  /* TODO: check for singularity and positive-semidefiniteness of cov */
  AIA_TENSOR_APPLY(T, s, *s_data = (*s_data < 0 ? 0 : *s_data););

  aiatensor__(sqrt)(s, s);
  aiatensor__(emulmv)(v, v, s);
  aiatensor__(mm)(res, tmp, v);
  aiatensor__(eaddmv)(res, res, mean);

  aiatensor__(free)(tmp);
  aiatensor__(free)(u);
  aiatensor__(free)(s);
  aiatensor__(free)(v);
}

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiarandom/random.c"
#include <aianon/core/erasure.h>