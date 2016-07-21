#ifndef AIA_RANDOM_SAMPLE_H

#include <aiarandom/generator.h>
#include <aiautil/util.h>
#include <aianon/tensor/tensor.h>

#ifndef PI
# define PI 3.14159265358979323846
#endif

#ifdef ERASED_TYPE_PRESENT

/* Generates a uniform random number between [a,b) */
AIA_API T aiarandom__(uniform)(AIARandGen *gen, T a, T b);

/* Generates a random number from a normal distribution
   with mean 'mean' and standard deviation 'stdv' >= 0 */
AIA_API T aiarandom__(normal)(AIARandGen *gen, T mean, T stdv);

/* Generates a random number from an exponential distribution.
   The density is 'p(x) = lambda * exp(-lambda * x)', where
   lambda is a positive number */
AIA_API T aiarandom__(exponential)(AIARandGen *gen, T lambda);

/* Returns a random number from a Cauchy distribution.
   The Cauchy density is 'p(x) = sigma / (pi * (sigma ^ 2 + (x - median) ^ 2))' */
AIA_API T aiarandom__(cauchy)(AIARandGen *gen, T median, T sigma);

/* Generates a random number from a log-normal distribution with
   'mean' is the mean of the log-normal distribution
   and 'stdv' > 0 is its standard deviation */
AIA_API T aiarandom__(logNormal)(AIARandGen *gen, T mean, T stdv);

/* Generates a random number from a geometric distribution.
   It returns an integer 'i', where 'p(i) = (1 - p) * p ^ (i - 1)'.
   p must satisfy '0 < p < 1' */
AIA_API int aiarandom__(geometric)(AIARandGen *gen, T p);

/* Returns true with probability 'p' and false with probability '1 - p' (p > 0) */
AIA_API int aiarandom__(bernoulli)(AIARandGen *gen, T p);

#endif

#ifndef aiarandom
#define aiarandom_(type, name) AIA_FN_ERASE_(random, type, name)
#define aiarandom__(name) aiarandom_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiarandom/sample.h"
#include <aiautil/erasure.h>

#define AIA_RANDOM_SAMPLE_H
#endif
