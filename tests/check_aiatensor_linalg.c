#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>

static float rndpd4x4[16] =
  { 0.63392f,  0.92338f,  0.88542f,  1.07961f,
    0.92338f,  2.11086f,  1.57509f,  2.06504f,
    0.88542f,  1.57509f,  1.71206f,  2.13114f,
    1.07961f,  2.06504f,  2.13114f,  2.79528f };

static float rndpd4x4L[16] =
  { 0.796190f,  0.000000f,  0.000000f,  0.000000f,
    1.159746f,  0.875126f,  0.000000f,  0.000000f,
    1.112069f,  0.326090f,  0.607474f,  0.000000f,
    1.355968f,  0.562729f,  0.723828f,  0.340642f };

static float rndb4x2[8] =
  { 0.022674f,  0.948729f,
    0.863286f,  0.363155f,
    0.491413f,  0.989033f,
    0.079586f,  0.187567f };

static long size4x4[2] = {4l, 4l};
static long size4x2[2] = {4l, 2l};

static float fepsi = 1e-4f;

AIATensor(float) *fpdtnsrc;
AIATensor(float) *fpdLtnsrc;
AIATensor(float) *ftnsrb;

// result tensors
AIATensor(float) *frestnsr;
AIATensor(float) *fexptnsr;

void tensorlinalg_setup(void) {
  fpdtnsrc  = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4, 16), 2, size4x4, NULL);
  fpdLtnsrc = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4L, 16), 2, size4x4, NULL);
  ftnsrb = aiatensor_(float, newFromData)(arr_(float, clone)(rndb4x2, 8), 2, size4x2, NULL);
}

void tensorlinalg_teardown(void) {
  aiatensor_(float, free)(fpdtnsrc);
  aiatensor_(float, free)(fpdLtnsrc);
  aiatensor_(float, free)(ftnsrb);
}

START_TEST(test_potrf_float) {
  frestnsr = aiatensor_(float, new)(fpdtnsrc);

  aiatensor_(float, potrf)(frestnsr, fpdtnsrc, "L");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fpdLtnsrc, fepsi),
    "potrf test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fpdLtnsrc), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_potrs_float) {
  float exp4x2[8] =
    { -3.3458f,  1.5483f,
       2.0851f,  0.0408f,
       7.3088f,  8.5890f,
      -5.7919f, -7.1093f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x2, 16), 2, size4x2, NULL);
  frestnsr = aiatensor_(float, newCopy)(ftnsrb);

  aiatensor_(float, potrs)(frestnsr, ftnsrb, fpdLtnsrc, "L");
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "wrong size for output");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr),
      "wrong size for output %dx%d", frestnsr->size[0], frestnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "potrs test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

Suite *make_tensorlinalg_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("TensorLinAlg");
  tc = tcase_create("TensorLinAlg");

  tcase_add_checked_fixture(tc, tensorlinalg_setup, tensorlinalg_teardown);

  tcase_add_test(tc, test_potrf_float);
  tcase_add_test(tc, test_potrs_float);

  suite_add_tcase(s, tc);
  return s;
}