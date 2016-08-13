#include <aiaoptim/optim.h>
#include <aiautil/util.h>
#include <aiatensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

CGConfig_ default_(cg_config) = {
  .gradtol = asT(1e-5),
  .maxIter = 100,
  .ls_config = &default_(ls_config),
  .ls = optim__(lsbacktrack)
};

void optim__(cg)(AIATensor_ *x, optim__(opfunc) opfunc, AIATensor_ *H, void *opstate, CGConfig_ *config) {
  if(!config) config = &default_(cg_config);

  AIATensor_ *rk  = aiatensor__(emptyVector)(H->size[0]);
  AIATensor_ *pk  = aiatensor__(emptyVector)(x->size[0]);
  AIATensor_ *Hpk = aiatensor__(empty)();

  T beta, alpha, rkTrk, rkTrkold, pkTHpk;
  long k = 0;

  opfunc(x, NULL, rk, ONLY_GRAD, opstate);

  // p_0 = - r_0
  aiatensor__(copy)(pk, rk);
  aiatensor__(mul)(pk, pk, -1);

  // r_k.T * r_k
  rkTrk = aiatensor__(dot)(rk, rk);

  while((rkTrk > config->gradtol) && (k < config->maxIter)) {
    // alpha_k = r_k.T * r_k / p_k.T * H * p_k
    aiatensor__(mv)(Hpk, H, pk);
    pkTHpk  = aiatensor__(dot)(Hpk, pk);
    alpha = rkTrk / pkTHpk;
    // x_k+1 = x_k + alpha_k * p_k
    aiatensor__(cadd)(x, x, alpha, pk);
    // r_k+1 = r_k + alpha_k * H * p_k
    aiatensor__(cadd)(rk, rk, alpha, Hpk);
    // copy r_k.T * r_k to rkTrkold and recompute r_k.T * r_k
    rkTrkold = rkTrk;
    rkTrk = aiatensor__(dot)(rk, rk);
    // beta_k+1 = r_k+1.T * r_k+1 / r_k.T * r_k
    beta = rkTrk/ rkTrkold;
    // p_k+1 = - r_k+1 + beta_k+1 * p_k
    aiatensor__(mul)(pk, pk, beta);
    aiatensor__(csub)(pk, pk, 1, rk);
    k++;
  }

  aiatensor__(free)(rk);
  aiatensor__(free)(pk);
  aiatensor__(free)(Hpk);
}

void optim__(ncg)(AIATensor_ *x, optim__(opfunc) opfunc, void *opstate, CGConfig_ *config) {
  if(!config) config = &default_(cg_config);

  AIATensor_ *gfc = aiatensor__(emptyAs)(x);
  AIATensor_ *gfx = aiatensor__(emptyAs)(x);
  AIATensor_ *pk  = aiatensor__(emptyAs)(x);

  T f, dgc, dgx, dgxc, alpha, beta;
  long k = 0;
  int lsresp;

  opfunc(x, &f, gfx, F_N_GRAD, opstate);
  aiatensor__(copy)(gfc, gfx);

  // pk = - gfc
  AIA_TENSOR_APPLY2(T, pk, T, gfc, *pk_data = - *gfc_data;);

  // gfc.T * gfc
  dgc = aiatensor__(dot)(gfc, gfc);

  while((dgc > config->gradtol) && (k < config->maxIter)) {
    // copy old values
    dgx = dgc;
    SWAP(AIATensor_*, gfc, gfx);
    // compute alpha_k
    alpha = 1 / (1 + dgc);
    lsresp = config->ls(&alpha, x, &f, gfc, opfunc, opstate, pk, NULL, NULL, gfx, NULL);
    if(lsresp == -1) break;
    // compute new dgc gfc.T * gfc and dgxc gfc.T * gfx
    dgc = 0;
    dgxc = 0;
    AIA_TENSOR_APPLY2(T, gfc, T, gfx,
                      dgc += *gfc_data * *gfc_data;
                      dgxc += *gfc_data * *gfx_data;
                      );
    // compute beta_k+1 = fprime_k+1.T * (fprime_k+1 - fprime_k) / fprime_k * fprime_k
    beta = (dgc - dgxc) / dgx;
    // compute p_k+1 = - fprime_k+1 + beta_k+1 * p_k
    AIA_TENSOR_APPLY2(T, pk, T, gfc, *pk_data = - *gfc_data + beta * *pk_data;);
    k++;
  }
  aiatensor__(free)(gfc);
  aiatensor__(free)(gfx);
  aiatensor__(free)(pk);
}

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/cg.c"
#include <aiautil/erasure.h>