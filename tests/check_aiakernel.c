#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiakernel/kernel.h>

static float rndveca4[4] =
  {0.2f,  0.9f,  0.5f,  0.4f};

static float rndvecb4[4] =
  {0.3f,  0.1f,  0.3f, 0.8f};

static float lamdiag4x4[4] =
  {0.99f,  0.19f,  0.76f,  0.45f};

static float datax3x4[12] =
  { 0.86135f,  0.31974f,  0.80091f,  0.17351f,
    0.99017f,  0.35123f,  0.38303f,  0.28982f,
    0.11373f,  0.99950f,  0.17728f,  0.32502f };

static float datay2x4[8] =
  { 0.78660f,  0.67510f,  0.67119f,  0.20291f,
    0.46390f,  0.63175f,  0.20450f,  0.93239f };

static float lamndiag4x4[16] =
  { 1.8145f,  1.4170f,  1.6234f,  1.1068f,
    1.4170f,  1.5718f,  0.9793f,  0.6321f,
    1.6234f,  0.9793f,  1.8274f,  1.1215f,
    1.1068f,  0.6321f,  1.1215f,  0.8097f };

static float lamndiag4x4L[16] =
  { 1.34703f,  0.00000f,  0.00000f,  0.00000f,
    1.05194f,  0.68207f,  0.00000f,  0.00000f,
    1.20516f, -0.42292f,  0.44284f,  0.00000f,
    0.82165f, -0.34048f, -0.02875f,  0.13350f };

static float alpha = 0.4f;

static long size4x4[2] = {4l, 4l};
static long size3x4[2] = {3l, 4l};
static long size3x3[2] = {3l, 3l};
static long size2x4[2] = {2l, 4l};
static long size3x2[2] = {3l, 2l};
static long size4[1]   = {4l};

static float fepsi = 1e-5f;

AIATensor(float) *fvec1;
AIATensor(float) *fvec2;
AIATensor(float) *fdatax;
AIATensor(float) *fdatay;
AIATensor(float) *flamdiag;
AIATensor(float) *flamndiagL;
AIATensor(float) *frestnsr;
AIATensor(float) *fexptnsr;

void kernel_setup(void) {
  fvec1      = aiatensor_(float, newFromData)(arr_(float, clone)(rndveca4, 4), 1, size4, NULL);
  fvec2      = aiatensor_(float, newFromData)(arr_(float, clone)(rndvecb4, 4), 1, size4, NULL);
  fdatax     = aiatensor_(float, newFromData)(arr_(float, clone)(datax3x4, 12), 2, size3x4, NULL);
  fdatay     = aiatensor_(float, newFromData)(arr_(float, clone)(datay2x4, 9), 2, size2x4, NULL);
  flamdiag   = aiatensor_(float, newFromData)(arr_(float, clone)(lamdiag4x4, 4), 1, size4, NULL);
  flamndiagL = aiatensor_(float, newFromData)(arr_(float, clone)(lamndiag4x4L, 16), 2, size4x4, NULL);
}

void kernel_teardown(void) {
  aiatensor_(float, free)(fvec1);
  aiatensor_(float, free)(fvec2);
  aiatensor_(float, free)(fdatax);
  aiatensor_(float, free)(fdatay);
  aiatensor_(float, free)(flamdiag);
  aiatensor_(float, free)(flamndiagL);
}

START_TEST(test_sekernel_matrix_float) {
  frestnsr = aiatensor_(float, empty)();
  // x and y are different and lambda is diagonal
  float exp3x2_1[6] =
    { 0.113069f, 0.047717f,
      0.111630f, 0.069998f,
      0.080843f, 0.069893f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp3x2_1, 12), 2, size3x2, NULL);

  aiakernel_se_(float, matrix)(frestnsr, fdatax, fdatay, 0.4f, flamdiag, TRUE, NULL);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sekernel_matrix test 1 failed. actual result = \n%s and expected result = \n%s",
    aiatensor_(float, mat2str)(frestnsr), aiatensor_(float, mat2str)(fexptnsr));
  aiatensor_(float, free)(fexptnsr);

  // y is NULL and lambda is diagonal
  float exp3x3_1[9] =
    { 0.160000f, 0.138971f, 0.026991f,
      0.138971f, 0.160000f, 0.034885f,
      0.026991f, 0.034885f, 0.160000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp3x3_1, 12), 2, size3x3, NULL);

  aiakernel_se_(float, matrix)(frestnsr, fdatax, NULL, 0.4f, flamdiag, TRUE, NULL);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sekernel_matrix test 2 failed. actual result = \n%s and expected result = \n%s",
    aiatensor_(float, mat2str)(frestnsr), aiatensor_(float, mat2str)(fexptnsr));
  aiatensor_(float, free)(fexptnsr);

  // x and y are different and lambda is non-diagonal
  float exp3x2_2[6] =
    { 0.010669f, 0.000000f,
      0.001322f, 0.000000f,
      0.000000f, 0.0294814f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp3x2_2, 12), 2, size3x2, NULL);

  aiakernel_se_(float, matrix)(frestnsr, fdatax, fdatay, 0.4f, flamndiagL, FALSE, "L");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sekernel_matrix test 3 failed. actual result = \n%s and expected result = \n%s",
    aiatensor_(float, mat2str)(frestnsr), aiatensor_(float, mat2str)(fexptnsr));
  aiatensor_(float, free)(fexptnsr);

  // y is NULL and lambda is non-diagonal
  float exp3x3_2[9] =
    { 0.160000f, 0.065816f, 0.000000f,
      0.065816f, 0.160000f, 0.000000f,
      0.000000f, 0.000000f, 0.160000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp3x3_2, 12), 2, size3x3, NULL);

  aiakernel_se_(float, matrix)(frestnsr, fdatax, NULL, 0.4f, flamndiagL, FALSE, "L");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sekernel_matrix test 4 failed. actual result = \n%s and expected result = \n%s",
    aiatensor_(float, mat2str)(frestnsr), aiatensor_(float, mat2str)(fexptnsr));

  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_sekernel_value_float) {
  float exp, res;

  // x and y are different and lambda is diagonal
  exp = 0.024090f;
  res = aiakernel_se_(float, value)(fvec1, fvec2, alpha, flamdiag, TRUE, NULL);
  ck_assert_msg(fabsf(res - exp) <= fepsi, "sekernel_value test failed. expected value = %f and actual value = %f", exp, res);

  // y is NULL and lambda is diagonal
  exp = 0.16f;
  res = aiakernel_se_(float, value)(fvec1, NULL, alpha, flamdiag, TRUE, NULL);
  ck_assert_msg(fabsf(res - exp) <= fepsi, "sekernel_value test failed. expected value = %f and actual value = %f", exp, res);

  // x and y are different and lambda is non-diagonal
  exp = 0.006109f;
  res = aiakernel_se_(float, value)(fvec1, fvec2, alpha, flamndiagL, FALSE, "L");
  ck_assert_msg(fabsf(res - exp) <= fepsi, "sekernel_value test failed. expected value = %f and actual value = %f", exp, res);

  // y is NULL and lambda is non-diagonal
  exp = 0.16f;
  res = aiakernel_se_(float, value)(fvec1, NULL, alpha, flamndiagL, FALSE, "L");
  ck_assert_msg(fabsf(res - exp) <= fepsi, "sekernel_value test failed. expected value = %f and actual value = %f", exp, res);
}
END_TEST


Suite *make_kernel_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("Kernel");
  tc = tcase_create("Kernel");

  tcase_add_checked_fixture(tc, kernel_setup, kernel_teardown);

  tcase_add_test(tc, test_sekernel_value_float);
  tcase_add_test(tc, test_sekernel_matrix_float);

  suite_add_tcase(s, tc);
  return s;
}