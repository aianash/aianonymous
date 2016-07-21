#include <aiaoptim/optim.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

adagrad_config default_adagrad_config = {
  .learningRate = 1e-3f,
  .learningRateDecay = 0.0f,
  .weightDecay = 0.0f,
};

#endif

#ifdef ERASED_TYPE_PRESENT

AIATensor_ *optim__(adagrad)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, adagrad_config *config, optim_state_(adagrad) *state) {
  aia_argcheck(state == NULL, 5, "state parameter expected got null");
  if(!config) config = &default_adagrad_config;

  float lr  = config->learningRate;
  float lrd = config->learningRateDecay;
  float wd  = config->weightDecay;

  int nevals = state->evalCounter;

  T fx;
  AIATensor_ *df_dx = aiatensor__(emptyAs)(x);
  opfunc(x, &fx, df_dx, F_N_GRAD);

  if(wd != 0.0)
    aiatensor__(cadd)(df_dx, df_dx, wd, x);

  float clr = lr / (1 + nevals * lrd);

  if(!state->paramVariance) {
    state->paramVariance = aiatensor__(emptyAs)(df_dx);
    aiatensor__(zero)(state->paramVariance);
  }
  aiatensor__(addcmul)(state->paramVariance, state->paramVariance, 1, df_dx, df_dx);
  state->paramStd = aiatensor__(newCopy)(state->paramVariance);
  aiatensor__(sqrt)(state->paramStd, state->paramStd);
  aiatensor__(add)(state->paramStd, state->paramStd, 1e-10);
  aiatensor__(addcdiv)(x, x, -clr, df_dx, state->paramStd);

  state->evalCounter++;

  *fx_ = fx;
  return x;
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/adagrad.c"
#include <aiautil/erasure.h>