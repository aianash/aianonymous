#include <aiaoptim/optim.h>
#include <aiautil/util.h>
#include <aiatensor/tensor.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

cg_config default_cg_config = {
  .gradtol = 1e-5f,
  .maxiter = 100
};

#endif

#ifdef ERASED_TYPE_PRESENT

void optim__(cg)(AIATensor_ *x, optim__(opfunc) opfunc, AIATensor_ *H, void *opstate, cg_config *config) {
  if(!config) config = &default_cg_config;

  AIATensor_ *rk  = aiatensor__(emptyVector)(H->size[0]);
  AIATensor_ *pk  = aiatensor__(emptyVector)(x->size[0]);
  AIATensor_ *Hpk = aiatensor__(empty)();

  T beta, alpha, rkTrk, pkTHpk;
  long k = 0;

  opfunc(x, NULL, rk, ONLY_GRAD, opstate);

  // p_0 = - r_0
  aiatensor__(copy)(pk, rk);
  aiatensor__(mul)(pk, pk, -1);

  // r_k.T * r_k
  rkTrk = aiatensor__(dot)(rk, rk);

  while((rkTrk > config->gradtol) && (k < config->maxiter)) {
    // alpha_k = r_k.T * r_k / p_k.T * H * p_k
    aiatensor__(mv)(Hpk, H, pk);
    pkTHpk  = aiatensor__(dot)(Hpk, pk);
    alpha = rkTrk / pkTHpk;
    // x_k+1 = x_k + alpha_k * p_k
    aiatensor__(cadd)(x, x, alpha, pk);
    // r_k+1 = r_k + alpha_k * H * p_k
    aiatensor__(cadd)(rk, rk, alpha, Hpk);
    // beta_k+1 = r_k+1.T * r_k+1 / r_k.T * r_k
    beta = aiatensor__(dot)(rk, rk)/ rkTrk;
    // p_k+1 = - r_k+1 + beta_k+1 * p_k
    aiatensor__(mul)(pk, pk, beta);
    aiatensor__(csub)(pk, pk, 1, rk);
    // r_k+1.T * r_k+1
    rkTrk = aiatensor__(dot)(rk, rk);
    k++;
  }

  aiatensor__(free)(rk);
  aiatensor__(free)(pk);
  aiatensor__(free)(Hpk);
}

void optim__(ncg)(AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, cg_config *config) {
  if(!config) config = &default_cg_config;

  AIATensor_ *df_dx     = aiatensor__(emptyAs)(x);
  AIATensor_ *df_dx_old = aiatensor__(emptyAs)(x);
  AIATensor_ *pk        = aiatensor__(emptyAs)(x);

  T f, grad, gradold, alpha, beta;
  long k = 0;

  opfunc(x, &f, df_dx, F_N_GRAD, opstate);
  // initialization
  aiatensor__(copy)(pk, df_dx);
  aiatensor__(mul)(pk, pk, -1);

  // df_dx.T * df_dx
  grad = aiatensor__(dot)(df_dx, df_dx);

  while((grad > config->gradtol) && (k < config->maxiter)) {
    // copy old values
    gradold = grad;
    aiatensor__(copy)(df_dx_old, df_dx);
    // compute alpha_k
    alpha = 1 / (1 + grad);
    optim__(lsbacktrack)(&alpha, opfunc, opstate, x, pk, &f, df_dx, NULL);
    // compute new grad
    grad = aiatensor__(dot)(df_dx, df_dx);
    // beta_k+1 = fprime_k+1.T * (fprime_k+1 - fprime_k) / fprime_k * fprime_k
    beta = (grad - aiatensor__(dot)(df_dx, df_dx_old)) / gradold;
    // p_k+1 = - fprime_k+1 + beta_k+1 * p_k
    aiatensor__(mul)(pk, pk, beta);
    aiatensor__(csub)(pk, pk, 1, df_dx);
    k++;
  }

  aiatensor__(free)(df_dx);
  aiatensor__(free)(df_dx_old);
  aiatensor__(free)(pk);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/cg.c"
#include <aiautil/erasure.h>