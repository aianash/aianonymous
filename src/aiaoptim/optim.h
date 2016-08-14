#ifndef AIANON_OPTIM_H

#include <aiautil/util.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

typedef enum { ONLY_F, ONLY_GRAD, F_N_GRAD } opfunc_ops;
typedef enum { LS_WOLFE_ARMIJO, LS_WOLFE_WEAK_CURVATURE, LS_WOLFE_STRONG_CURVATURE } wolfe_condition;

// Line Search return states
enum {
  LS_SUCCESS = 0,
  LS_CONVERGENCE = 0,

  LSERR_UNKNOWN = -1024,    // probably logic error, consider as fatal error
  LSERR_MAX_ITER,           // max iteration for line search reached
  LSERR_MIN_STEP,           // min step length reached
  LSERR_MAX_STEP,           // max step length reached
  LSERR_INVALID_PARAM,      // invalid parameters
  LSERR_INVALID_DIR_GRAD,   // directional gradient is positive
  LSERR_MAXMIN_STEP_WITHIN_TOL, // diff between max and min of step length within tolerance
  LSERR_ROUNDING_ERR,           // rounding error caused step length to overflow beyond max and min

  // More Thuente linesearch specific
  LSMTERR_INVALID_TRIAL,    // invalid trial value for ls more thuente
};

AIA_API char *lserr2str(int errno);

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

extern sgd_config default_sgd_config;
extern adagrad_config default_adagrad_config;

#endif


#ifdef ERASED_TYPE_PRESENT

// OpFunc, which returns grad and value of the function to be minimized at
// given x
typedef void (*optim__(opfunc))(AIATensor_ *x, T *fx, AIATensor_ *df_dx, opfunc_ops ops, void *opstate);


/*************************** LINE SEARCH ****************************/

typedef struct LSConfig_ {
  int maxIter;
  T c1;
  T c2;
  T dec;
  T inc;
  T amax;
  T amin;
  T xtol;
  wolfe_condition wolfe;
} LSConfig_;

typedef int (*optim__(ls))(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa,
                          optim__(opfunc) opfunc, void *opstate,
                          AIATensor_ *p, AIATensor_ *x, T *fx, AIATensor_ *gfx,
                          LSConfig_ *config);

extern LSConfig_ default_(ls_config);

/** Linear search routines **/

/**
 * Description
 * -----------
 *
 * Input
 * -----
 * a      : Initial Estimate of step length
 * opfunc : Function to optimize
 * p      : Search direction
 * x      : Initial value of x. xa is used as initial value if NULL
 * f      : Function value at x. fa is used as initial function value if NULL
 * gf     : Gradient at x. gfa is used as initial function value if NULL
 * config : Line search config. Default config is used if NULL
 *
 * Output
 * ------
 * a      : Final step length
 * xa     : xa = x + a * p
 * fa     : Function value at xa
 * gfa    : Gradient at xa
 *
 * Returns
 * -------
 * Error or success, refer LS error codes above
 *
 * References
 * ----------
 * .. [1] J. Nocedal and S. Wright, *Numerical Optimization*
 * .. [2] Jorge J. More and David J. Thuente, *Line search algorithm with guaranteed sufficient decrease*,
 *        ACM Transactions on Mathematical Software (TOMS), Vol. 20, No. 3, pp. 286-307, 1994.
 */
AIA_API int optim__(lsmorethuente)(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa,
                                  optim__(opfunc) opfunc, void *opstate,
                                  AIATensor_ *p, AIATensor_ *x, T *f, AIATensor_ *gf,
                                  LSConfig_ *config);

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
 * Error or success, refer LS error codes above
 *
 * References
 * ----------
 * .. [1] J. Nocedal and S. Wright, *Numerical Optimization*
 */
AIA_API int optim__(lsbacktrack)(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa,
                                optim__(opfunc) opfunc, void *opstate,
                                AIATensor_ *p, AIATensor_ *x, T *fx, AIATensor_ *gfx,
                                LSConfig_ *config);


/************************* Conjugate Gradient *************************/

typedef struct CGConfig_ {
  T gradtol;
  long maxIter;
  LSConfig_ *ls_config;
  optim__(ls) ls;
} CGConfig_;

extern CGConfig_ default_(cg_config);

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
AIA_API void optim__(cg)(AIATensor_ *x, optim__(opfunc) opfunc, AIATensor_ *H, void *opstate, CGConfig_ *config);

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
AIA_API void optim__(ncg)(AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, CGConfig_ *config);


/********************************** Stochastic Gradient ********************************/

typedef struct optim_state_(sgd) {
  AIATensor_ *df_dx;
  int evalCounter;
} optim_state_(sgd);

typedef struct optim_state_(adagrad) {
  AIATensor_ *paramVariance;
  AIATensor_ *paramStd;
  int evalCounter;
} optim_state_(adagrad);

AIA_API AIATensor_ *optim__(sgd)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, sgd_config *config, optim_state_(sgd) *state);
AIA_API AIATensor_ *optim__(adagrad)(T *fx_, AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, adagrad_config *config, optim_state_(adagrad) *state);

#endif

#ifndef optim_state
#define optim_state(type, method) AIA_CONCAT_3(optim_state, method, type)
#define optim_state_(method) optim_state(T_, method)
#endif

#ifndef optim_
#define optim_(type, method) AIA_FN_ERASE_(optim, type, method)
#define optim__(method) optim_(T_, method)
#endif

#ifndef CGConfig_
#define CGConfig(type) AIA_CONCAT_3(CGConfig, _, type)
#define CGConfig_ CGConfig(T_)
#endif

#ifndef LSConfig_
#define LSConfig(type) AIA_CONCAT_3(LSConfig, _, type)
#define LSConfig_ LSConfig(T_)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/optim.h"
#include <aiautil/erasure.h>

#define AIANON_OPTIM_H
#endif