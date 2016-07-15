#ifndef AIANON_OPTIM_H

#include <aianon/core/util.h>
#include <aianon/tensor/tensor.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

typedef struct sgd_config_
{
  float learningRate;
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

typedef struct state__(sgd)
{
  AIATensor_ *df_dx;
  AIATensor_ *decayParameters;
  AIATensor_ *deltaParameters;
  int evalCounter;
} state__(sgd);

typedef void (*optim__(opfunc))(AIATensor_ *x, T *fx, AIATensor_ *df_dx);

AIA_API void optim__(sgd)(AIATensor_ *xx, AIATensor_ *fx, optim__(opfunc) opfunc, AIATensor_ *x, sgd_config *config, state__(sgd) *state);

#endif

#ifndef state_
#define state_(type, method) AIA_CONCAT_3(state_, sgd_, type)
#define state__(method) state_(T_, method)
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