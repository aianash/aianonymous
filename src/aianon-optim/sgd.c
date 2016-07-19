#include <aianon-optim/optim.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

sgd_config default_sgd_config = {
  .learningRate = 1e-3f,
  .learningRateDecay = 0.0f,
  .weightDecay = 0.0f,
  .momentum = 0.0f,
  .dampening = 0.0f,
  .nesterov = 0,
  .weightDecays = NULL,
  .learningRates = NULL
};

#endif

#ifdef ERASED_TYPE_PRESENT

AIATensor_ *optim__(sgd)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, sgd_config *config, optim_state_(sgd) *state) {
  aia_argcheck(state == NULL, 5, "state parameter expected got null");
  if(!config) config = &default_sgd_config;

  float damp     = config->dampening;
  float mom      = config->momentum;
  float wd       = config->weightDecay;
  float lr       = config->learningRate;
  float lrd      = config->learningRateDecay;
  float nesterov = config->nesterov;
  int nevals     = state->evalCounter;

  float *wds = config->weightDecays;
  float *lrs = config->learningRates;

  // evaluate f(x) and df/dx
  T fx;
  AIATensor_ *df_dx = aiatensor__(emptyAs)(x);
  opfunc(x, &fx, df_dx, F_N_GRAD);

  // apply weight decay with single or individual parameters
  if(wd != 0.0)
    aiatensor__(cadd)(df_dx, df_dx, wd, x);
  else if(wds) {
    AIATensor_ *wdsTnsr = aiatensor__(emptyAs)(x);
    aiatensor__(copyFloat)(wdsTnsr, wds);
    aiatensor__(addcmul)(df_dx, df_dx, 1, wdsTnsr, x);
    aiatensor__(free)(wdsTnsr);
  }

  // apply momentum
  if(mom != 0) {
    if(!state->df_dx)
      state->df_dx = aiatensor__(newCopy)(df_dx);
    else {
      aiatensor__(mul)(state->df_dx, state->df_dx, mom);
      aiatensor__(cadd)(state->df_dx, state->df_dx, (1 - damp), df_dx);
    }
    if(nesterov)
      aiatensor__(cadd)(df_dx, df_dx, mom, state->df_dx);
    else
      aiatensor__(copy)(df_dx, state->df_dx);
  }

  // learning rate decay (annealing)
  float clr = lr / (1 + nevals * lrd);

  // parameter update (single or individual learning rate)
  if(lrs) {
    AIATensor_ *lrsTnsr = aiatensor__(emptyAs)(x);
    aiatensor__(copyFloat)(lrsTnsr, lrs);
    aiatensor__(addcmul)(x, x, -clr, lrsTnsr, df_dx);
    aiatensor__(free)(lrsTnsr);
  } else
    aiatensor__(cadd)(x, x, -clr, df_dx);

  state->evalCounter += 1;

  *fx_ = fx;
  return x;
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon-optim/sgd.c"
#include <aianon/core/erasure.h>