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
    k += 1;
  }

  aiatensor__(free)(rk);
  aiatensor__(free)(pk);
  aiatensor__(free)(Hpk);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/cg.c"
#include <aiautil/erasure.h>