#ifndef AIANON_OPTIM_H

#include <aianon/core/util.h>
#include <aianon/tensor/tensor.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

typedef struct sgd_config_ {
  float learningRate;
  float *learningRates;
  float learningRateDecay;
  float weightDecay;
  float *weightDecays;
  float momentum;
  float dampening;
  int nesterov;
} sgd_config;

extern sgd_config default_sgd_config;

#endif

#ifdef ERASED_TYPE_PRESENT

typedef struct optim_state_(sgd) {
  AIATensor_ *df_dx;
  int evalCounter;
} optim_state_(sgd);

typedef void (*optim__(opfunc))(AIATensor_ *x, T *fx, AIATensor_ *df_dx);

AIA_API AIATensor_ *optim__(sgd)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, sgd_config *config, optim_state_(sgd) *state);

#endif

#ifndef optim_state
#define optim_state(type, method) AIA_CONCAT_3(optim_state, sgd_, type)
#define optim_state_(method) optim_state(T_, method)
#endif

#ifndef optim_
#define optim_(type, method) AIA_FN_ERASE_(optim, type, method)
#define optim__(method) optim_(type, method)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon-optim/optim.h"
#include <aianon/core/erasure.h>

#define AIANON_OPTIM_H
#endif