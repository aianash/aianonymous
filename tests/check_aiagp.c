#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiagp/gp.h>

/**
 * Testcases not complete
 */

static float fK6x6[36] =
  { 1.00000f,  0.88692f,  0.61878f,  0.47236f,  0.33959f,  0.22992f,
    0.88692f,  1.00000f,  0.88692f,  0.76337f,  0.61878f,  0.47236f,
    0.61878f,  0.88692f,  1.00000f,  0.97044f,  0.88692f,  0.76337f,
    0.47236f,  0.76337f,  0.97044f,  1.00000f,  0.97044f,  0.88692f,
    0.33959f,  0.61878f,  0.88692f,  0.97044f,  1.00000f,  0.97044f,
    0.22992f,  0.47236f,  0.76337f,  0.88692f,  0.97044f,  1.00000f };

static float fK6x6L[36] =
  { 1.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,
    0.88692f,  0.46192f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,
    0.61878f,  0.73196f,  0.28520f,  0.00000f,  0.00000f,  0.00000f,
    0.47236f,  0.74564f,  0.46413f,  0.07393f,  0.00000f,  0.00000f,
    0.33959f,  0.68753f,  0.60845f,  0.20245f,  0.02757f,  0.00000f,
    0.22992f,  0.58113f,  0.68629f,  0.35779f,  0.10118f,  0.01234f };

static float fy6[6] =
  { 0.84147f,  0.42336f, -4.79462f, -1.67649f,  4.59890f,  7.91486f };

static long size6x6[2] = {6l, 6l};
static long size6x4[2] = {6l, 4l};
static long size4x4[2] = {4l, 4l};
static long size6[1]   = {6l};
static long size4x1[1] = {4l};

AIATensor(float) *fK6x6Ltnsr;
AIATensor(float) *fytnsr;
AIATensor(float) *fK6x6tnsr;

static float fepsi = 1e-5f;

void gp_setup(void) {
  fK6x6tnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fK6x6, 36), 2, size6x6, NULL);
  fK6x6Ltnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fK6x6L, 36), 2, size6x6, NULL);
  fytnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fy6, 6), 1, size6, NULL);
}

void gp_teardown(void) {
  aiatensor_(float, free)(fK6x6Ltnsr);
}

START_TEST(test_spredc_float) {
  float Kx6[6] = { 0.934517f,  0.992602f,  0.829340f,  0.692827f,  0.545079f,  0.403865f };
  AIATensor(float) *Kx = aiatensor_(float, newFromData)(arr_(float, clone)(Kx6, 6), 1, size6, NULL);
  AIATensor(float) *beta = aiagp_(float, calcbeta)(NULL, fK6x6Ltnsr, LOWER_MAT, fytnsr);
  float Kxx = 1;
  // printf("beta = %s\n", aiatensor_(float, toString)(beta));
  float fmean, fcov;
  aiagp_(float, spredc)(&fmean, &fcov, fK6x6Ltnsr, LOWER_MAT, Kx, Kxx, beta);
  // printf("fmean = %f\tfcov = %f\n", fmean, fcov);
  aiatensor_(float, free)(Kx);
  aiatensor_(float, free)(beta);
}
END_TEST

START_TEST(test_npredc_float) {
  float Kx6x4[24] =
    { 0.97044f,  0.84930f,  0.38161f,  0.08803f,
      0.76337f,  0.99667f,  0.66808f,  0.22992f,
      0.47236f,  0.92004f,  0.92004f,  0.47236f,
      0.33959f,  0.80788f,  0.98675f,  0.61878f,
      0.22992f,  0.66808f,  0.99667f,  0.76337f,
      0.14660f,  0.52030f,  0.94806f,  0.88692f };
  float Kxx4x4[16] =
    { 1.00000f,  0.71653f,  0.26359f,  0.04978f,
      0.71653f,  1.00000f,  0.71653f,  0.26359f,
      0.26359f,  0.71653f,  1.00000f,  0.71653f,
      0.04978f,  0.26359f,  0.71653f,  1.00000f };
  AIATensor(float) *Kx  = aiatensor_(float, newFromData)(arr_(float, clone)(Kx6x4, 24), 2, size6x4, NULL);
  AIATensor(float) *Kxx = aiatensor_(float, newFromData)(arr_(float, clone)(Kxx4x4, 24), 2, size4x4, NULL);
  AIATensor(float) *beta = aiagp_(float, calcbeta)(NULL, fK6x6Ltnsr, LOWER_MAT, fytnsr);
  //printf("beta = \n%s\n", aiatensor_(float, toString)(beta));
  AIATensor(float) *fmean = aiatensor_(float, empty)();
  AIATensor(float) *fcov  = aiatensor_(float, empty)();

  aiagp_(float, npredc)(fmean, fcov, fK6x6Ltnsr, LOWER_MAT, Kx, Kxx, beta);
  //printf("fmean = \n%s\n", aiatensor_(float, toString)(fmean));
  //printf("fcov = \n%s\n", aiatensor_(float, toString)(fcov));

  aiatensor_(float, free)(Kx);
  aiatensor_(float, free)(Kxx);
  aiatensor_(float, free)(beta);
  aiatensor_(float, free)(fmean);
  aiatensor_(float, free)(fcov);
}
END_TEST

START_TEST(test_opfuncse_float) {
  // anisotropic kernel
  float exp6x1[6] =
    { 0.62480f, -0.14161f,  -0.00215f,  -0.00849f,  -0.03384f,  -0.00746f };
  long size6x1[1] = {6l};
  AIATensor(float) *fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp6x1, 6), 1, size6x1, NULL);
  float fexpres = 4.00342f;

  float *fres = malloc(sizeof(float));
  AIATensor(float) *frestnsr = aiatensor_(float, empty)();

  float datax3x4[12] =
    { 0.86135f,  0.31974f,  0.80091f,  0.17351f,
      0.99017f,  0.35123f,  0.38303f,  0.28982f,
      0.11373f,  0.99950f,  0.17728f,  0.32502f };
  long size3x4[2] = {3l, 4l};
  AIATensor(float) *fdataxtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(datax3x4, 12), 2, size3x4, NULL);

  float datay3x1[3] =
    { 0.86135f,  0.31974f,  0.80091f};
  long size3x1[1] = {3l};
  AIATensor(float) *fdataytnsr = aiatensor_(float, newFromData)(arr_(float, clone)(datay3x1, 3), 1, size3x1, NULL);

  float x6x1[6] =
    { 0.6f, 0.4f, 0.99f,  0.19f,  0.76f,  0.45f};
  AIATensor(float) *fxtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(x6x1, 6), 1, size6x1, NULL);

  // initialize gp state
  AIAGpState(float) *state = malloc(sizeof(AIAGpState(float)));
  state->X = fdataxtnsr;
  state->y = fdataytnsr;
  state->isokernel = FALSE;

  opfunc_ops ops = F_N_GRAD;

  aiagp_(float, opfuncse)(fxtnsr, fres, frestnsr, ops, state);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "opfuncse test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));
  ck_assert_msg(epsieqf(*fres, fexpres, fepsi),
    "opfuncse test failed.\nexpected output =\n%0.5f\nactual output =\n%0.5f\n",
    fexpres, *fres);

  free(fres);
  free(state);
  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fdataxtnsr);
  aiatensor_(float, free)(fdataytnsr);
  aiatensor_(float, free)(fxtnsr);
}
END_TEST

START_TEST(test_cggpse_float) {
  AIATensor(float) *lambda = aiatensor_(float, empty)();
  AIATensor(float) *K = aiatensor_(float, empty)();
  AIATensor(float) *Kx = aiatensor_(float, empty)();
  AIATensor(float) *Kxx = aiatensor_(float, empty)();
  AIATensor(float) *KPS = aiatensor_(float, empty)();
  AIATensor(float) *KPSchol = aiatensor_(float, empty)();
  AIATensor(float) *beta = aiatensor_(float, empty)();
  AIATensor(float) *fmean = aiatensor_(float, empty)();
  AIATensor(float) *fcov  = aiatensor_(float, empty)();

  float sigma, alpha;
  float *ele;

  // anisotropic kernel
  float datax3x4[12] =
    { 0.86135f,  1.31974f,  0.80091f,  0.17351f,
      0.99017f,  0.35123f,  1.38303f,  0.28982f,
      0.11373f,  1.99950f,  0.17728f,  1.32502f };
  long size3x4[2] = {3l, 4l};
  AIATensor(float) *fdataxtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(datax3x4, 12), 2, size3x4, NULL);

  float datay3x1[3] =
    { 1.86135f,  0.01974f,  2.10091f};
  long size3x1[1] = {3l};
  AIATensor(float) *fdataytnsr = aiatensor_(float, newFromData)(arr_(float, clone)(datay3x1, 3), 1, size3x1, NULL);

  float datatest4x4[32] =
    { 0.74135f,  1.67974f,  0.78091f,  0.37351f,
      0.85017f,  0.45123f,  1.33303f,  0.54982f,
      0.97373f,  0.90950f,  0.76728f,  1.12502f,
      0.85017f,  1.45123f,  0.33303f,  0.54982f,
      0.97373f,  0.90950f,  1.76728f,  0.12502f,
      0.85017f,  0.45123f,  0.33303f,  1.54982f,
      0.97373f,  1.90950f,  0.76728f,  0.12502f,
      0.92373f,  0.39950f,  1.61728f,  0.76502f };
  long size8x4[2] = {8l, 4l};
  AIATensor(float) *fdatatesttnsr = aiatensor_(float, newFromData)(arr_(float, clone)(datatest4x4, 32), 2, size8x4, NULL);

  sigma = 0.2f; // use for pred
  alpha = 0.4f; // use for pred
  //sigma = 0.01241f; // use for test
  //alpha = 0.70425f; // use for test

  float x4x1[4] = {0.7f,  5.f, 2.f,  4.f}; // use for pred
  //float x4x1[4] = {0.68084f,  5.10886f, 2.42689f, 7.76903f}; // use for test
  lambda = aiatensor_(float, newFromData)(arr_(float, clone)(x4x1, 4), 1, size4x1, NULL);
  ele = NULL;
  vforeach(ele, lambda) {
    *ele = *ele * *ele;
  }
  endvforeach()

  // covariance of training inputs
  aiakernel_se_(float, matrix)(K, fdataxtnsr, NULL, alpha, lambda, DIAG_MAT);
  // convert to k + sigma ^ 2 * I
  aiatensor_(float, aEyepX)(KPS, K, sigma * sigma);
  // calculate cholskey decomposition of KPS
  aiatensor_(float, resizeAs)(KPSchol, KPS);
  aiatensor_(float, potrf)(KPSchol, KPS, LOWER_MAT);
  // calculate beta
  aiagp_(float, calcbeta)(beta, KPSchol, LOWER_MAT, fdataytnsr);
  // cross covariance test and training inputs
  aiakernel_se_(float, matrix)(Kx, fdataxtnsr, fdatatesttnsr, alpha, lambda, DIAG_MAT);
  // covariance of test inputs
  aiakernel_se_(float, matrix)(Kxx, fdatatesttnsr, NULL, alpha, lambda, DIAG_MAT);
  // calculate mean and covar of prediction
  aiagp_(float, npredc)(fmean, fcov, KPSchol, LOWER_MAT, Kx, Kxx, beta);
  //printf("fmean = %s\n", aiatensor_(float, toString)(fmean));
  //printf("fcov = %s\n", aiatensor_(float, toString)(fcov));

  float dataypred4x1[8] =
    { 1.40152f, 0.84430f, 0.96919f, 1.37042f, 0.49383f, 1.22233f, 1.06897f, 0.58308f };
  long size8x1[1] = {8l};
  AIATensor(float) *fdataypredtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(dataypred4x1, 8), 1, size8x1, NULL);

  // initialize gp state
  AIAGpState(float) *state = malloc(sizeof(AIAGpState(float)));
  state->X = fdatatesttnsr;
  state->y = fdataypredtnsr;
  state->isokernel = FALSE;

  // initialize x
  float x6x1[6] = { 1.f, 1.0f, 1.0f,  4.0f,  1.0f,  7.0f};
  long size6x1[1] = {6l};
  AIATensor(float) *fxtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(x6x1, 6), 1, size6x1, NULL);

  optim_(float, ncg)(fxtnsr, aiagp_(float, opfuncse), state, &default_cg_config);
  //printf("fxtnsr = %s\n", aiatensor_(float, toString)(fxtnsr));

  free(state);
  aiatensor_(float, free)(fdataxtnsr);
  aiatensor_(float, free)(fdataytnsr);
  aiatensor_(float, free)(fdatatesttnsr);
  aiatensor_(float, free)(fdataypredtnsr);
  aiatensor_(float, free)(fxtnsr);
  aiatensor_(float, free)(lambda);
  aiatensor_(float, free)(K);
  aiatensor_(float, free)(Kx);
  aiatensor_(float, free)(Kxx);
  aiatensor_(float, free)(KPS);
  aiatensor_(float, free)(KPSchol);
  aiatensor_(float, free)(beta);
  aiatensor_(float, free)(fmean);
  aiatensor_(float, free)(fcov);
}
END_TEST


Suite *make_gp_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("GaussianProcess");
  tc = tcase_create("GaussianProcess");

  tcase_add_checked_fixture(tc, gp_setup, gp_teardown);

  tcase_add_test(tc, test_spredc_float);
  tcase_add_test(tc, test_npredc_float);

  tcase_add_test(tc, test_opfuncse_float);
  tcase_add_test(tc, test_cggpse_float);

  suite_add_tcase(s, tc);
  return s;
}