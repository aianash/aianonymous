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

AIATensor(float) *fK6x6Ltnsr;
AIATensor(float) *fytnsr;

void gp_setup(void) {
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
  printf("beta = \n%s\n", aiatensor_(float, toString)(beta));
  AIATensor(float) *fmean = aiatensor_(float, empty)();
  AIATensor(float) *fcov  = aiatensor_(float, empty)();
  aiagp_(float, npredc)(fmean, fcov, fK6x6Ltnsr, LOWER_MAT, Kx, Kxx, beta);
  printf("fmean = \n%s\n", aiatensor_(float, toString)(fmean));
  printf("fcov = \n%s\n", aiatensor_(float, toString)(fcov));
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

  suite_add_tcase(s, tc);
  return s;
}