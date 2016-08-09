#ifndef AIANON_OPTIM_H

#include <aiautil/util.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

typedef enum { ONLY_F, ONLY_GRAD, F_N_GRAD } opfunc_ops;
typedef enum { LS_WOLFE_ARMIJO, LS_WOLFE_WEAK_CURVATURE, LS_WOLFE_STRONG_CURVATURE } wolfe_condition;

/** configs **/

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

typedef struct adagrad_config_ {
  float learningRate;
  float learningRateDecay;
  float weightDecay;
} adagrad_config;

typedef struct cg_config_ {
  float gradtol;
  long maxiter;
} cg_config;

typedef struct ls_config_ {
  int maxiter;
  float c1;
  float c2;
  float dec;
  float inc;
  float amax;
  float amin;
  wolfe_condition wolfe;
} ls_config;

extern sgd_config default_sgd_config;
extern adagrad_config default_adagrad_config;
extern cg_config default_cg_config;

#endif

#ifdef ERASED_TYPE_PRESENT

typedef struct optim_state_(sgd) {
  AIATensor_ *df_dx;
  int evalCounter;
} optim_state_(sgd);

typedef struct optim_state_(adagrad) {
  AIATensor_ *paramVariance;
  AIATensor_ *paramStd;
  int evalCounter;
} optim_state_(adagrad);

typedef void (*optim__(opfunc))(AIATensor_ *x, T *fx, AIATensor_ *df_dx, opfunc_ops ops, void *opstate);

/** Stochastic Gradient **/

AIA_API AIATensor_ *optim__(sgd)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, sgd_config *config, optim_state_(sgd) *state);
AIA_API AIATensor_ *optim__(adagrad)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, adagrad_config *config, optim_state_(adagrad) *state);

/**
 * Description
 * -----------
 * Conjugate Gradient algorithm to solve minima of function of form
 *   f(x) = x.T * A * x + b.T * x
 *   where A is symmetric positive definite matrix
 *
 * Input
 * -----
 * x       : Initial guess of mimina
 * opfunc  : Function to optimize
 * H       : Hessian matrix of f(x)
 * opstate : State of opfunc
 * config  : cg_config
 *
 * Output
 * ------
 * x       : Minima of f(x)
 */
AIA_API void optim__(cg)(AIATensor_ *x, optim__(opfunc) opfunc, AIATensor_ *H, void *opstate, cg_config *config);

/**
 * Description
 * -----------
 * Non-linear Conjugate Gradient algorithm to find local minima of a function.
 * This algorithm uses Polak-Ribiere variant.
 *
 * Input
 * -----
 * x       : Initial guess of mimina
 * opfunc  : Function to optimize
 * opstate : State of opfunc
 * config  : cg_config
 *
 * Output
 * ------
 * x       : Minima of f(x)
 */
AIA_API void optim__(ncg)(AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, cg_config *config);

/** Linear search routines **/

/**
 * Description
 * -----------
 *
 * Input
 * -----
 * a        :
 * opfunc   :
 * x        :
 * p        :
 * f        :
 * gf       :
 * c1       :
 * c2       :
 * amin     :
 * amax     :
 * xtol     :
 * maxIter  :
 *
 * Output
 * ------
 * a    :
 * f    :
 * gf   :
 * ////
 */
AIA_API int optim__(lsmorethuente)(T *a, optim__(opfunc) opfunc, void *opstate, AIATensor_ *x, AIATensor_ *p, T *f, AIATensor_ *gf, T c1, T c2, T amax, T amin, T xtol, int maxIter);

/**
 * Description
 * -----------
 * Backtracking line search algorithm
 *
 * Input
 * -----
 * a       : Initial estimate of step length
 * opfunc  : Function to optimize
 * opstate : Function state
 * p       : Search direction
 * x       : Initial value of x. xa is used as initial value if NULL
 * fx      : Function value at x. fa is used as initial function value if NULL
 * gfx     : Gradient at x. gfa is used as initial function value if NULL
 * config  : Line search config. Default config is used if NULL
 *
 * Output
 * ------
 * a       : Step length
 * xa      : xa = x + a * p
 * fa      : Function value at xa
 * gfa     : Gradient at xa
 *
 * Returns
 * -------
 * -1 in case of error, positive value otherwise
 *
 * References
 * ----------
 * 1. Numerical Optimization, J. Nocedal and S. Wright
 */
AIA_API int optim__(lsbacktrack)(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa, optim__(opfunc) opfunc, void *opstate, AIATensor_ *p, AIATensor_ *x, T *fx, AIATensor_ *gfx, ls_config *config);

#endif

#ifndef optim_state
#define optim_state(type, method) AIA_CONCAT_3(optim_state, method, type)
#define optim_state_(method) optim_state(T_, method)
#endif

#ifndef optim_
#define optim_(type, method) AIA_FN_ERASE_(optim, type, method)
#define optim__(method) optim_(T_, method)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/optim.h"
#include <aiautil/erasure.h>

#define AIANON_OPTIM_H
#endif