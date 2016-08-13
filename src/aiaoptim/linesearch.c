#include <aiaoptim/optim.h>

#ifdef ERASED_TYPE_PRESENT

LSConfig_ default_(ls_config) = {
  .maxIter = 20,
  .c1      = asT(0.05),
  .c2      = asT(0.1),
  .dec     = asT(0.5),
  .inc     = asT(2.1),
  .amax    = asT(1e+20),
  .amin    = asT(1e-20),
  .xtol    = asT(1e-5),
  .wolfe   = LS_WOLFE_STRONG_CURVATURE
};

//
static int optim__(zoom)(T *ax, T *fxm, T *dgxm, T *ay, T *fym, T *dgym, T *ac, T *fcm, T *dgcm, int *brackt, T acmin, T acmax);

// TODO : Major recoding required
int optim__(lsmorethuente)(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa,
                          optim__(opfunc) opfunc, void *opstate,
                          AIATensor_ *p, AIATensor_ *x, T *f, AIATensor_ *gf,
                          LSConfig_ *config) {
  if(!config) config = &default_(ls_config);
  if(!x) x = xa;
  if(!f) f = fa;
  if(!gf) gf = gfa;

  int fcount;
  int iter;
  int brackt; // = 1 means minimizer has been bracketed
  int zerr;
  int stage1;
  T amax = config->amax;
  T amin = config->amin;
  T c1 = config->c1;
  T c2 = config->c2;
  T xtol = config->xtol;
  long maxIter = config->maxIter;
  T width, pwidth;
  T ax, fx, dgx; // values at the best step
  T ay, fy, dgy; // values at the other endpoint of the interval of uncertainity

  T ac, fc, dgc; // values at current step
  AIATensor_ *xc; // x at current step
  AIATensor_ *gfc; // gradient of function at current step
  T acmin, acmax;

  T fxm, dgxm, fym, dgym, fcm, dgcm;
  T phi0, dphi0;
  T la, c1dphi0;

  if(*a <= 0) return LSERR_INVALID_PARAM;

  iter = 0;
  fcount = 0;
  brackt = FALSE;
  width = amax - amin;
  pwidth = 2.0 * width;

  phi0 = *f;
  dphi0 = aiatensor__(dot)(p, gf);
  if(dphi0 > 0) return LSERR_INVALID_DIR_GRAD;

  c1dphi0 = c1 * dphi0;

  xc = aiatensor__(emptyAs)(x);
  gfc = aiatensor__(emptyAs)(x);
  ac = *a;
  fc = phi0;

  ax = ay = 0.;
  fx = fy = phi0;
  dgx = dgy = dphi0;

  while(1) {

    if(brackt) {
      acmin = min2(ax, ay);
      acmax = min2(ax, ay);
    } else {
      acmin = ax;
      acmax = ac + 4.0 * (ac - ax);
    }

    // clip current step to the provided max min of step
    if(ac < amin) ac = amin;
    if(ac > amax) ac = amax;

    // for unusual termination set current step
    // to the lowest obtained so far
    if((brackt && ((ac <= acmin || ac >= acmax) || iter + 1 >= maxIter || zerr != 0)) ||
       (brackt && (acmax - acmin <= xtol * acmax))) {
      *a = ax;
    }

    // compute current x
    // xc = x + ac * p
    aiatensor__(cadd)(xc, x, ac, p);

    // Evaluate function and gradients at current x
    opfunc(xc, &fc, gfc, F_N_GRAD, opstate);
    dgc = aiatensor__(dot)(p, gfc);
    fcount++;

    // new l(alpha) for current step
    la = phi0 + ac * c1dphi0;

    if(brackt && ((ac <= acmin || ac >= acmax) || zerr != 0)) {
      return LSERR_ROUNDING_ERR;
    }
    if(ac == amax && fc <= la && dgc <= c1dphi0) {
      *a = ac;
      return LSERR_MAX_STEP;
    }
    if(ac == amin && (fc > la || c1dphi0 <= dgc)) {
      *a = ac;
      return LSERR_MIN_STEP;
    }
    if(brackt && (acmax - acmin <= xtol * acmax)) {
      return LSERR_MAXMIN_STEP_WITHIN_TOL;
    }
    if(iter <= maxIter) {
      return LSERR_MAX_ITER;
    }

    // converged to step satisfying
    // sufficient decrease condition and curvature condition
    if(fc <= la && fabs(dgc) <= c2 * (-dphi0)) {
      *a = ac;
      *f = fc;
      aiatensor__(freeCopyTo)(gfc, gf);
      return fcount;
    }

    if(stage1 && *f <= c1dphi0 && min2(c1, c2) * dphi0 <= dgc)
      stage1 = 0;

    if(stage1 && fc >= la && fc <= fx) {
      fcm = fc - ac * c1dphi0;
      fxm = fx - ax * c1dphi0;
      fym = fy - ay * c1dphi0;
      dgcm = dgc - c1dphi0;
      dgxm = dgx - c1dphi0;
      dgym = dgy - c1dphi0;

      zerr = optim__(zoom)(&ax, &fxm, &dgxm, &ay, &fym, &dgym, &ac, &fcm, &dgcm, &brackt, acmin, acmax);

      fx = fxm + ax * c1dphi0;
      fy = fym + ax * c1dphi0;
      dgx = dgxm + c1dphi0;
      dgy = dgym + c1dphi0;

    } else {
      zerr = optim__(zoom)(&ax, &fx, &dgx, &ay, &fy, &dgy, &ac, &fc, &dgc, &brackt, acmin, acmax);
    }

    if(brackt) {
      if(0.66 * pwidth <= fabs(ay - ax)) {
        ac = ax + 0.5 * (ay - ax);
      }
      pwidth = width;
      width = fabs(ay - ax);
    }
  }

  return LSERR_UNKNOWN;
}


/**
 * Define the local variables for computing minimizers.
 */
#define USES_MINIMIZER \
    T a, d, gamma, theta, p, q, r, s;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 */
#define CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = max3(p, q, r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s)); \
    if ((v) < (u)) gamma = -gamma; \
    p = gamma - (du) + theta; \
    q = gamma - (du) + gamma + (dv); \
    r = p / q; \
    (cm) = (u) + r * d;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 *  @param  xmin    The maximum value.
 *  @param  xmin    The minimum value.
 */
#define CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = max3(p, q, r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(max2(0, a * a - ((du) / s) * ((dv) / s))); \
    if ((u) < (v)) gamma = -gamma; \
    p = gamma - (dv) + theta; \
    q = gamma - (dv) + gamma + (du); \
    r = p / q; \
    if (r < 0. && gamma != 0.) { \
        (cm) = (v) - r * d; \
    } else if (a < 0) { \
        (cm) = (xmax); \
    } else { \
        (cm) = (xmin); \
    }

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 */
#define QUARD_MINIMIZER(qm, u, fu, du, v, fv) \
    a = (v) - (u); \
    (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  dv      The value of f'(v).
 */
#define QUARD_MINIMIZER2(qm, u, du, v, dv) \
    a = (u) - (v); \
    (qm) = (v) + (dv) / ((dv) - (du)) * a;

//
int optim__(zoom)(T *ax, T *fx, T *dgx, T *ay, T *fy, T *dgy, T *at, T *ft, T *dgt, int *brackt, T atmin, T atmax) {
  int bound;
  int dsign = fsigndiff(dgt, dgx);
  T mc, mq, atn;

  USES_MINIMIZER;

  if(*brackt) {
    if(*at <= min2(*ax, *ay) || max2(*ax, *ay) <= *at) {
      return LSMTERR_INVALID_TRIAL;
    }
    if(0. <= *dgx * (*at - *ax)) {
      return LSERR_INVALID_DIR_GRAD;
    }
    if(atmax < atmin) {
      return LSERR_INVALID_PARAM;
    }
  }

  // Trial value selection
  if(*fx < *ft) {
    // Case 1: a higher function value.
    // The minimum is brackt. If the cubic minimizer is closer
    // to x than the quadratic one, the cubic one is taken, else
    // the average of the minimizers is taken.
    *brackt = 1;
    bound = 1;
    CUBIC_MINIMIZER(mc, *ax, *fx, *dgx, *at, *ft, *dgt);
    QUARD_MINIMIZER(mq, *ax, *fx, *dgx, *at, *ft);
    if(fabs(mc - *ax) < fabs(mq - *ax)) {
      atn = mc;
    } else {
      atn = mc + 0.5 * (mq - mc);
    }
  } else if(dsign) {
    // Case 2: a lower function value and derivatives of
    // opposite sign. The minimum is brackt. If the cubic
    // minimizer is closer to x than the quadratic (secant) one,
    // the cubic one is taken, else the quadratic one is taken.
    *brackt = 1;
    bound = 0;
    CUBIC_MINIMIZER(mc, *ax, *fx, *dgx, *at, *ft, *dgt);
    QUARD_MINIMIZER2(mq, *ax, *dgx, *at, *dgt);
    if(fabs(mc - *at) > fabs(mq - *at)) {
      atn = mc;
    } else {
      atn = mq;
    }
  } else if(fabs(*dgt) < fabs(*dgx)) {
    // Case 3: a lower function value, derivatives of the
    // same sign, and the magnitude of the derivative decreases.
    // The cubic minimizer is only used if the cubic tends to
    // infinity in the direction of the minimizer or if the minimum
    // of the cubic is beyond t. Otherwise the cubic minimizer is
    // defined to be either tmin or tmax. The quadratic (secant)
    // minimizer is also computed and if the minimum is brackt
    // then the the minimizer closest to x is taken, else the one
    // farthest away is taken.
    bound = 1;
    CUBIC_MINIMIZER2(mc, *ax, *fx, *dgx, *at, *ft, *dgt, atmin, atmax);
    QUARD_MINIMIZER2(mq, *ax, *dgx, *at, *dgt);
    if(*brackt) {
      if(fabs(*at - mc) < fabs(*at - mq)) {
        atn = mc;
      } else {
        atn = mq;
      }
    } else {
      if(fabs(*at - mc) > fabs(*at - mq)) {
        atn = mc;
      } else {
        atn = mq;
      }
    }
  } else {
    // Case 4: a lower function value, derivatives of the
    // same sign, and the magnitude of the derivative does
    // not decrease. If the minimum is not brackt, the step
    // is either tmin or tmax, else the cubic minimizer is taken.
    bound = 0;
    if(*brackt) {
      CUBIC_MINIMIZER(atn, *at, *ft, *dgt, *ay, *fy, *dgy);
    } else if(*ax < *at) {
      atn = atmax;
    } else {
      atn = atmin;
    }
  }

  // Update the interval of uncertainty. This update does not
  // depend on the new step or the case analysis above.

  // - Case a: if f(x) < f(t),
  //     x <- x, y <- t.
  // - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
  //     x <- t, y <- y.
  // - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
  //     x <- t, y <- x.
  if(*fx < *ft) {
    // case a
    *ay = *at;
    *fy = *ft;
    *dgy = *dgt;
  } else {
    // case c
    if(dsign) {
      *ay = *ax;
      *fy = *fx;
      *dgy = *dgt;
    }
    // case b and c
    *ax = *at;
    *fx  = *ft;
    *dgx = *dgt;
  }

  if(atn > atmax) atn = atmax;
  if(atn < atmin) atn = atmin;

  // Redefine the new trial value if it is close to the upper bound
  // of the interval.
  if(*brackt && bound) {
    mq = *ax + 0.66 * (*ay - *ax);
    if(*ax < *ay) {
      if(atn > mq) atn = mq;
    } else {
      if(atn > mq) atn = mq;
    }
  }

  *at = atn;
  return LS_SUCCESS;
}

int optim__(lsbacktrack)(T *a, AIATensor_ *xa, T *fa, AIATensor_ *gfa, optim__(opfunc) opfunc, void *opstate, AIATensor_ *p, AIATensor_ *x, T *fx, AIATensor_ *gfx, LSConfig_ *config) {
  if(!config) config = &default_(ls_config);
  if(!x) x = xa;
  if(!fx) fx = fa;
  if(!gfx) gfx = gfa;

  T width, dga, dgx;
  T finit = *fx;
  T ap = 0;
  int count = 0;

  // check for errors
  if(*a <= 0) return -1;

  // check initial gradient in direction of p
  dgx = aiatensor__(dot)(gfx, p);
  if(dgx > 0) return -1;

  if(x != xa)
    aiatensor__(copy)(xa, x);

  while(count <= config->maxIter) {
    // x_i+1 = x_i + (a_i+1 - a_i) * p
    AIA_TENSOR_APPLY2(T, xa, T, p, *xa_data += - ap * *p_data + *a * *p_data; );

    // compute function value and gradient at new xa
    opfunc(xa, fa, gfa, F_N_GRAD, opstate);
    count++;

    if(*fa > finit + *a * config->c1 * dgx) {
      width = config->dec;
    } else {
      // if Armijo condition to be used
      if(config->wolfe == LS_WOLFE_ARMIJO) {
        return count;
      }
      // check for wolfe condition
      dga = aiatensor__(dot)(gfa, p);
      if(dga < config->c2 * dgx) {
        width = config->inc;
      } else {
        // if wolfe condition to be used
        if(config->wolfe == LS_WOLFE_WEAK_CURVATURE) {
          return count;
        }
        // check for strong wolfe condition
        if(dga > - config->c2 * dgx) {
          width = config->dec;
        } else {
          return count;
        }
      }
    }

    if(*a < config->amin) {
      return -1;
    }
    if(*a > config->amax) {
      return -1;
    }

    ap = *a;
    (*a) *= width;
  }
}


#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aiaoptim/linesearch.c"
#include <aiautil/erasure.h>