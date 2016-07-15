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
  .weightDecays = NULL
};

#endif

#ifdef ERASED_TYPE_PRESENT

void optim__(sgd)(AIATensor_ *xx, AIATensor_ *fx_, optim__(opfunc) opfunc, AIATensor_ *x, sgd_config *config, state__(sgd) *state) {

  // // evaluate f(x) and df/dx
  // T fx;
  // AIATensor_ *df_dx = aiatensor__(empty)();
  // opfunc(xx, &fx, df_dx);

}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon-optim/cg.c"
#include <aianon/core/erasure.h>