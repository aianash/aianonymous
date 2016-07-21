#ifndef AIA_TENSOR_RANDOM_H

#include <aiarandom/generator.h>
#include <aiarandom/sample.h>
#include <aiautil/util.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>

#ifdef ERASED_TYPE_PRESENT

AIA_API void aiatensor__(random)(AIATensor_ *res, AIARandGen *gen);
AIA_API void aiatensor__(geometric)(AIATensor_ *res, AIARandGen *gen, T p);
AIA_API void aiatensor__(bernoulli)(AIATensor_ *res, AIARandGen *gen, T p);
AIA_API void aiatensor__(uniform)(AIATensor_ *res, AIARandGen *gen, T a, T b);
AIA_API void aiatensor__(normal)(AIATensor_ *res, AIARandGen *gen, T mean, T stdv);
AIA_API void aiatensor__(exponential)(AIATensor_ *res, AIARandGen *gen, T lambda);
AIA_API void aiatensor__(cauchy)(AIATensor_ *res, AIARandGen *gen, T median, T sigma);
AIA_API void aiatensor__(logNormal)(AIATensor_ *res, AIARandGen *gen, T mean, T stdv);
/* multivariate normal */
AIA_API void aiatensor__(mvnormal)(AIATensor_ *res, AIARandGen *gen, AIATensor_ *mean, AIATensor_ *cov);

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiarandom/random.h"
#include <aiautil/erasure.h>

#define AIA_TENSOR_RANDOM_H
#endif