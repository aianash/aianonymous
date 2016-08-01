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

static float rnd5x5U[25] =
  { 6.80000f, -2.11000f,  5.66000f,  5.97000f,  8.23000f,
    0.00000f, -3.30000f,  5.36000f, -4.44000f,  1.08000f,
    0.00000f,  0.00000f, -2.70000f,  0.27000f,  9.04000f,
    0.00000f,  0.00000f,  0.00000f, -7.17000f,  2.14000f,
    0.00000f,  0.00000f,  0.00000f,  0.00000f, -6.87000f };

static float rndb4x2[8] =
  { 0.022674f,  0.948729f,
    0.863286f,  0.363155f,
    0.491413f,  0.989033f,
    0.079586f,  0.187567f };

static float rndc5x3[15] =
  { 4.02000f, -1.56000f,  9.81000f,
    6.19000f,  4.00000f, -4.09000f,
   -8.22000f, -8.67000f, -4.57000f,
   -7.57000f,  1.75000f, -8.61000f,
   -3.03000f,  2.86000f,  8.99000f };

static long size4x4[2] = {4l, 4l};
static long size4x2[2] = {4l, 2l};
static long size5x5[2] = {5l, 5l};
static long size5x3[2] = {5l, 3l};
static long size4[1] = {4l};

static float fepsi = 1e-4f;

AIATensor(float) *fpdtnsrc;
AIATensor(float) *fpdLtnsrc;
AIATensor(float) *ftnsrb;
AIATensor(float) *fUtnsrc;
AIATensor(float) *ftnsrcc;

// result tensors
AIATensor(float) *frestnsr;
AIATensor(float) *fexptnsr;

void tensorlinalg_setup(void) {
  fpdtnsrc  = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4, 16), 2, size4x4, NULL);
  fpdLtnsrc = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4L, 16), 2, size4x4, NULL);
  fUtnsrc = aiatensor_(float, newFromData)(arr_(float, clone)(rnd5x5U, 25), 2, size5x5, NULL);
  ftnsrb = aiatensor_(float, newFromData)(arr_(float, clone)(rndb4x2, 8), 2, size4x2, NULL);
  ftnsrcc = aiatensor_(float, newFromData)(arr_(float, clone)(rndc5x3, 15), 2, size5x3, NULL);
}

void tensorlinalg_teardown(void) {
  aiatensor_(float, free)(fpdtnsrc);
  aiatensor_(float, free)(fpdLtnsrc);
  aiatensor_(float, free)(fUtnsrc);
  aiatensor_(float, free)(ftnsrb);
  aiatensor_(float, free)(ftnsrcc);
}

START_TEST(test_potrf_float) {
  frestnsr = aiatensor_(float, new)(fpdtnsrc);

  aiatensor_(float, potrf)(frestnsr, fpdtnsrc, LOWER_MAT);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fpdLtnsrc, fepsi),
    "potrf test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fpdLtnsrc), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_potrs_float) {
  float exp4x2[8] =
    { -3.3458f,  1.5483f,
       2.0851f,  0.0408f,
       7.3088f,  8.5890f,
      -5.7919f, -7.1093f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x2, 8), 2, size4x2, NULL);
  frestnsr = aiatensor_(float, newCopy)(ftnsrb);

  aiatensor_(float, potrs)(frestnsr, ftnsrb, fpdLtnsrc, LOWER_MAT);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "wrong size for output");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr),
      "wrong size for output %dx%d", frestnsr->size[0], frestnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "potrs test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_trtrs_float) {
  float exp5x3[15] =
    { -3.54160f, -0.25140f,  3.08470f,
       4.20720f,  2.03910f, -4.51460f,
       4.63990f,  1.78040f, -2.60770f,
       1.18740f, -0.36830f,  0.81030f,
       0.44100f, -0.41630f, -1.30860f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp5x3, 15), 2, size5x3, NULL);
  AIATensor(float) *fresbtnsr = aiatensor_(float, newCopy)(ftnsrcc);

  aiatensor_(float, trtrs)(fresbtnsr, ftnsrcc, fUtnsrc, UPPER_MAT, "N", "N");
  ck_assert_msg(aiatensor_(float, isMatrix)(fresbtnsr), "wrong size for output");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresbtnsr, fexptnsr),
      "wrong size for output %dx%d", fresbtnsr->size[0], fresbtnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresbtnsr, fexptnsr, fepsi),
    "trtrs test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(fresbtnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(fresbtnsr);
}
END_TEST

START_TEST(test_svd_float) {
  float expu4x4[16] =
    { -0.27010f,  0.05520f, -0.91540f,  0.29340f
      -0.52320f,  0.82870f,  0.17300f, -0.09780f,
      -0.49580f, -0.37830f, -0.12390f, -0.77180f,
      -0.63830f, -0.40870f,  0.34170f,  0.55560f };

  float exps4[4] =
    {  6.59990f,  0.43490f,  0.17620f,  0.04110f };

  float expv4x4[16] =
    { -0.27010f,  0.05520f, -0.91540f,  0.29340f,
      -0.52320f,  0.82870f,  0.17300f, -0.09780f,
      -0.49580f, -0.37830f, -0.12390f, -0.77180f,
      -0.63830f, -0.40870f,  0.34170f,  0.55560f };

  AIATensor(float) *fexputnsr = aiatensor_(float, newFromData)(arr_(float, clone)(expv4x4, 16), 2, size4x4, NULL);
  AIATensor(float) *fexpstnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exps4, 4), 1, size4, NULL);
  AIATensor(float) *fexpvtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(expv4x4, 16), 2, size4x4, NULL);

  AIATensor(float) *fresutnsr = aiatensor_(float, emptyAs)(fexputnsr);
  AIATensor(float) *fresstnsr = aiatensor_(float, emptyVector)(size4[0]);
  AIATensor(float) *fresvtnsr = aiatensor_(float, emptyAs)(fexpvtnsr);

  aiatensor_(float, gesvd)(fresutnsr, fresstnsr, fresvtnsr, fpdtnsrc, "A");
  ck_assert_msg(aiatensor_(float, isMatrix)(fresutnsr), "wrong size for output u");
  ck_assert_msg(aiatensor_(float, isVector)(fresstnsr), "wrong size for output s");
  ck_assert_msg(aiatensor_(float, isMatrix)(fresvtnsr), "wrong size for output v");

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresutnsr, fexputnsr),
    "wrong size for output u %dx%d", fresutnsr->size[0], fresutnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresutnsr, fexputnsr, fepsi),
    "svd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexputnsr), aiatensor_(float, toString)(fresutnsr));

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresstnsr, fexpstnsr),
    "wrong size for output s %d", fresstnsr->size[0]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresstnsr, fexpstnsr, fepsi),
    "svd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexpstnsr), aiatensor_(float, toString)(fresstnsr));

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresvtnsr, fexpvtnsr),
    "wrong size for output v %dx%d", fresvtnsr->size[0], fresvtnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresvtnsr, fexpvtnsr, fepsi),
    "svd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexpvtnsr), aiatensor_(float, toString)(fresvtnsr));

  aiatensor_(float, free)(fexputnsr);
  aiatensor_(float, free)(fexpstnsr);
  aiatensor_(float, free)(fexpvtnsr);
  aiatensor_(float, free)(fresutnsr);
  aiatensor_(float, free)(fresstnsr);
  aiatensor_(float, free)(fresvtnsr);
}
END_TEST

START_TEST(test_syev_float) {
  float expv4x4[16] =
    {  0.29337f,  0.91539f,  0.05520f, 0.27007f,
      -0.09784f, -0.17297f,  0.82872f, 0.52318f,
      -0.77180f,  0.12387f, -0.37829f, 0.49583f,
       0.55558f, -0.34174f, -0.40872f, 0.63834f };

  float expe4[4] =
    {  0.04110f, 0.17620f, 0.43490f, 6.59990f };

  AIATensor(float) *fexpetnsr = aiatensor_(float, newFromData)(arr_(float, clone)(expe4, 4), 1, size4, NULL);
  AIATensor(float) *fexpvtnsr = aiatensor_(float, newFromData)(arr_(float, clone)(expv4x4, 16), 2, size4x4, NULL);

  AIATensor(float) *fresetnsr = aiatensor_(float, emptyVector)(size4[0]);
  AIATensor(float) *fresvtnsr = aiatensor_(float, emptyAs)(fexpvtnsr);

  aiatensor_(float, syev)(fresetnsr, fresvtnsr, fpdtnsrc, "V", UPPER_MAT);
  ck_assert_msg(aiatensor_(float, isVector)(fresetnsr), "wrong size for output e");
  ck_assert_msg(aiatensor_(float, isMatrix)(fresvtnsr), "wrong size for output v");

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresetnsr, fexpetnsr),
    "wrong size for output e %d", fresetnsr->size[0]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresetnsr, fexpetnsr, fepsi),
    "syev test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexpetnsr), aiatensor_(float, toString)(fresetnsr));

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(fresvtnsr, fexpvtnsr),
    "wrong size for output v %dx%d", fresvtnsr->size[0], fresvtnsr->size[1]);
  ck_assert_msg(aiatensor_(float, epsieq)(fresvtnsr, fexpvtnsr, fepsi),
    "syev test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexpvtnsr), aiatensor_(float, toString)(fresvtnsr));

  aiatensor_(float, free)(fexpetnsr);
  aiatensor_(float, free)(fexpvtnsr);
  aiatensor_(float, free)(fresetnsr);
  aiatensor_(float, free)(fresvtnsr);
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
  tcase_add_test(tc, test_trtrs_float);
  tcase_add_test(tc, test_svd_float);
  tcase_add_test(tc, test_syev_float);

  suite_add_tcase(s, tc);
  return s;
}