#include <aianon/random/sample.h>

#ifdef ERASED_TYPE_PRESENT

/* For pseudo random number sampling see https://en.wikipedia.org/wiki/Inverse_transform_sampling */

/* generates a random number between [0,1) */
static T aiarandom__(uniform01)(AIARandGen *gen) {
  #ifdef T_IS_DOUBLE  
    return (T)aiarandgen_double(gen);
  #elif defined(T_IS_FLOAT)
    return (T)aiarandgen_float(gen);
  #endif
}

T aiarandom__(uniform)(AIARandGen *gen, T a, T b) {
  return (aiarandom__(uniform01)(gen) * (b - a) + a);
}

T aiarandom__(normal)(AIARandGen *gen, T mean, T stdv) {
  aia_argcheck(stdv > 0, 2, "standard deviation should be positive");

  /* Box - Muller method */
  /* check https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform for details */
  if (!gen->isNormalValid) {
    gen->normalX = aiarandom__(uniform01)(gen);
    gen->normalY = aiarandom__(uniform01)(gen);
    gen->normalRho = sqrt(-2.0 * log(1.0 - gen->normalY));
    gen->isNormalValid = 1;
    return gen->normalRho * cos(2.0 * PI * gen->normalX) * stdv + mean;
  } else {
    gen->isNormalValid = 0;
    return gen->normalRho * sin(2.0 * PI * gen->normalX) * stdv + mean;
  }
}

/* For pseudo random number sampling see https://en.wikipedia.org/wiki/Inverse_transform_sampling */
T aiarandom__(exponential)(AIARandGen *gen, T lambda) {
  aia_argcheck(lambda > 0, 1, "lambda should be positive");
  return (-1.0 / lambda * log(1.0 - aiarandom__(uniform01)(gen)));
}

T aiarandom__(cauchy)(AIARandGen *gen, T median, T sigma) {
  return (median + sigma * tan(PI * (aiarandom__(uniform01)(gen) - 0.5)));
}

/* https://en.wikipedia.org/wiki/Log-normal_distribution */
T aiarandom__(logNormal)(AIARandGen *gen, T mean, T stdv) {
  aia_argcheck(stdv > 0, 2, "standard deviation should be positive");
  T m = mean * mean;
  T s = stdv * stdv;
  return (exp(aiarandom__(normal)(gen, log(m / sqrt(m + s)), sqrt(log(s / m + 1)))));
}

int aiarandom__(geometric)(AIARandGen *gen, T p) {
  aia_argcheck(p > 0 && p < 1, 1, "p must be between (0, 1)");
  return ((int)(log(1 - aiarandom__(uniform01)(gen)) / log(p)) + 1);
}

int aiarandom__(bernoulli)(AIARandGen *gen, T p) {
  aia_argcheck(p >= 0 && p <= 1, 1, "p must be between [0, 1]");
  return (aiarandom__(uniform01)(gen) <= p);
}

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/random/sample.c"
#include <aianon/core/erasure.h>