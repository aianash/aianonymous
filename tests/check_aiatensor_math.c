#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>
#include <aiautil/math.h>

static float rnda4x4[16] =
  { 0.44783f,  0.06268f,  0.03410f,  0.65443f,
    0.96924f,  0.85259f,  0.08037f,  0.66187f,
    0.93053f,  0.21757f,  0.59805f,  0.66421f,
    0.95874f,  0.55492f,  0.87603f,  0.89484f };

static float rnda4x4T[16] =
  { 0.44783f,  0.96924f,  0.93053f,  0.95874f,
    0.06268f,  0.85259f,  0.21757f,  0.55492f,
    0.03410f,  0.08037f,  0.59805f,  0.87603f,
    0.65443f,  0.66187f,  0.66421f,  0.89484f };

static float rndb4x4[16] =
  { 0.07991f,  0.97586f,  0.26686f,  0.91247f,
    0.84907f,  0.66531f,  0.93157f,  0.71106f,
    0.66486f,  0.30385f,  0.21018f,  0.77947f,
    0.41738f,  0.90427f,  0.16466f,  0.09119f };

static float rndb4x4T[16] =
  { 0.07991f,  0.84907f,  0.66486f,  0.41738f,
    0.97586f,  0.66531f,  0.30385f,  0.90427f,
    0.26686f,  0.93157f,  0.21018f,  0.16466f,
    0.91247f,  0.71106f,  0.77947f,  0.09119f };

static float rndc4x4[16] =
  { 0.2f, 0.1f, 0.3f, 0.8f,
    0.1f, 0.5f, 0.9f, 0.3f,
    0.7f, 0.5f, 0.6f, 0.1f,
    0.4f, 0.3f, 0.9f, 0.1f };

static float rndc4x4T[16] =
  { 0.2f,  0.1f,  0.7f,  0.4f,
    0.1f,  0.5f,  0.5f,  0.3f,
    0.3f,  0.9f,  0.6f,  0.9f,
    0.8f,  0.3f,  0.1f,  0.1f };

static float rndd4x4[16] =
  { 0.44783f,  0.06268f,  0.03410f,  0.65443f,
    0.96924f,  0.85259f,  2.08037f,  0.66187f,
    1.93053f,  0.21757f,  0.59805f,  -3.66421f,
    0.95874f,  -1.55492f,  0.87603f,  0.89484f };

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

static float rndveca4[4] =
  {0.2f,  0.9f,  0.5f,  0.4f};

static float rndvecb4[4] =
  {0.218616f,  0.974018f ,  0.820125f,  0.941869f};

static long size4x4[2] = {4l, 4l};
static long size4[1]   = {4l};
static long size4x1[2] = {4l, 1l};

static long stride4x1[2] = {4l, 1l};

static float fepsi = 5e-5f;

/** contiguous tensors */
AIATensor(float) *ftnsr1c;    // contiguous float tensor 1
AIATensor(float) *ftnsr2c;    // contiguous float tensor 2
AIATensor(float) *ftnsr3c;    // contiguous float tensor 3
AIATensor(float) *ftnsr4c;    // contiguous float tensor 4
AIATensor(float) *fpdtnsrc;   // positive definite float tensor
AIATensor(float) *fpdLtnsrc;  // cholesky decomposition (lower) of fpdtnsrc

/** transposed tensors */
AIATensor(float) *ftnsr1cT;   // transpose of ftnsr1c

/** non-contiguous tensors */
AIATensor(float) *ftnsr1nc;   // non-contiguous version of ftnsr1c

/** vectors */
AIATensor(float) *fvec1;      // float vector 1
AIATensor(float) *fvec2;      // float vector 2

/** result tensors */
AIATensor(float) *frestnsr;   // actual result
AIATensor(float) *fexptnsr;   // expected result

void tensormath_setup(void) {
  // matrix
  ftnsr1c   = aiatensor_(float, newFromData)(arr_(float, clone)(rnda4x4, 16), 2, size4x4, NULL);
  ftnsr2c   = aiatensor_(float, newFromData)(arr_(float, clone)(rndb4x4, 16), 2, size4x4, NULL);
  ftnsr3c   = aiatensor_(float, newFromData)(arr_(float, clone)(rndc4x4, 16), 2, size4x4, NULL);
  ftnsr4c   = aiatensor_(float, newFromData)(arr_(float, clone)(rndd4x4, 16), 2, size4x4, NULL);
  fpdtnsrc  = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4, 16), 2, size4x4, NULL);
  fpdLtnsrc = aiatensor_(float, newFromData)(arr_(float, clone)(rndpd4x4L, 16), 2, size4x4, NULL);
  ftnsr1cT  = aiatensor_(float, newFromData)(arr_(float, clone)(rnda4x4T, 16), 2, size4x4, NULL);
  ftnsr1nc  = aiatensor_(float, new)(ftnsr1cT);
  aiatensor_(float, transpose)(ftnsr1nc, ftnsr1cT, 0, 1);

  // vector
  fvec1 = aiatensor_(float, newFromData)(arr_(float, clone)(rndveca4, 4), 1, size4, NULL);
  fvec2 = aiatensor_(float, newFromData)(arr_(float, clone)(rndvecb4, 4), 1, size4, NULL);
}

void tensormath_teardown(void) {
  aiatensor_(float, free)(ftnsr1c);
  aiatensor_(float, free)(ftnsr2c);
  aiatensor_(float, free)(ftnsr3c);
  aiatensor_(float, free)(ftnsr4c);
  aiatensor_(float, free)(fpdtnsrc);
  aiatensor_(float, free)(fpdLtnsrc);
  aiatensor_(float, free)(ftnsr1cT);
  aiatensor_(float, free)(ftnsr1nc);
  aiatensor_(float, free)(fvec1);
  aiatensor_(float, free)(fvec2);
}

START_TEST(test_add_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, add)(frestnsr, ftnsr1c, 0.8327f);
  float exp4x4[16] =
    { 1.28053f,  0.89538f,  0.86680f,  1.48713f,
      1.80194f,  1.68529f,  0.91307f,  1.49457f,
      1.76323f,  1.05027f,  1.43075f,  1.49691f,
      1.79144f,  1.38762f,  1.70873f,  1.72754f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "add test failed. actual result = %s and expected result = %s",
    aiatensor_(float, toString)(frestnsr), aiatensor_(float, toString)(fexptnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_sub_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, sub)(frestnsr, ftnsr1c, 0.8327f);
  float exp4x4[16] =
    { -0.38487f, -0.77002f, -0.79860f, -0.17827f,
       0.13654f,  0.01989f, -0.75233f, -0.17083f,
       0.09783f, -0.61513f, -0.23465f, -0.16849f,
       0.12604f, -0.27778f,  0.04333f,  0.06214f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sub test failed. actual result = %s and expected result = %s",
    aiatensor_(float, toString)(frestnsr), aiatensor_(float, toString)(fexptnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_mul_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, mul)(frestnsr, ftnsr1c, 0.8f);
  float exp4x4[16] =
    { 0.358264f,  0.050144f,  0.027280f,  0.523544f,
      0.775392f,  0.682072f,  0.064296f,  0.529496f,
      0.744424f,  0.174056f,  0.478440f,  0.531368f,
      0.766992f,  0.443936f,  0.700824f,  0.715872f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "mul test failed. actual result = %s and expected result = %s",
    aiatensor_(float, toString)(frestnsr), aiatensor_(float, toString)(fexptnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_div_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, div)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 2.23915f,  0.31340f,  0.17050f,  3.27215f,
      4.84620f,  4.26295f,  0.40185f,  3.30935f,
      4.65265f,  1.08785f,  2.99025f,  3.32105f,
      4.79370f,  2.77460f,  4.38015f,  4.47420f  };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "div test failed. actual result = %s and expected result = %s",
    aiatensor_(float, toString)(frestnsr), aiatensor_(float, toString)(fexptnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_fmod_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, fmod)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.05443f,
      0.16924f,  0.05259f,  0.08037f,  0.06187f,
      0.13053f,  0.01757f,  0.19805f,  0.06421f,
      0.15874f,  0.15492f,  0.07603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "fmod test failed");

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_remainder_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, remainder)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.05443f,
      0.16924f,  0.05259f,  0.08037f,  0.06187f,
      0.13053f,  0.01757f,  0.19805f,  0.06421f,
      0.15874f,  0.15492f,  0.07603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "remainder test failed");

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_clamp_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, clamp)(frestnsr, ftnsr1c, 0.17328f, 0.51223f);
  float exp4x4[16] =
    { 0.44783f,  0.17328f,  0.17328f,  0.51223f,
      0.51223f,  0.51223f,  0.17328f,  0.51223f,
      0.51223f,  0.21757f,  0.51223f,  0.51223f,
      0.51223f,  0.51223f,  0.51223f,  0.51223f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "clamp test failed");

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cadd_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.471803f,  0.355438f,  0.114158f,  0.928171f,
      1.223961f,  1.052183f,  0.359841f,  0.875188f,
      1.129988f,  0.308725f,  0.661104f,  0.898051f,
      1.083954f,  0.826201f,  0.925428f,  0.922197f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when ftnsr1c and frestnsr are contiguous and different
  aiatensor_(float, cadd)(frestnsr, ftnsr1c, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when ftnsr1c and frestnsr are contiguous and same
  aiatensor_(float, copy)(frestnsr, ftnsr1c);
  aiatensor_(float, cadd)(frestnsr, frestnsr, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cadd)(frestnsr, ftnsr1nc, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_csub_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.423857, -0.230078, -0.045958,  0.380689,
      0.714519,  0.652997, -0.199101,  0.448552,
      0.731072,  0.126415,  0.534996,  0.430369,
      0.833526,  0.283639,  0.826632,  0.867483 };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when ftnsr1c and frestnsr are contiguous and different
  aiatensor_(float, csub)(frestnsr, ftnsr1c, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "csub test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when ftnsr1c and frestnsr are contiguous and same
  aiatensor_(float, copy)(frestnsr, ftnsr1c);
  aiatensor_(float, csub)(frestnsr, frestnsr, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "csub test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, csub)(frestnsr, ftnsr1nc, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cmul_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.089566f,  0.006268f,  0.010230f,  0.523544f,
      0.096924f,  0.426295f,  0.072333f,  0.198561f,
      0.651371f,  0.108785f,  0.358830f,  0.066421f,
      0.383496f,  0.166476f,  0.788427f,  0.089484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cmul)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cmul)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cpow_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.851574f,  0.758076f,  0.362932f,  0.712345f,
      0.996880f,  0.923358f,  0.103415f,  0.883551f,
      0.950848f,  0.466443f,  0.734585f,  0.959910f,
      0.983287f,  0.838048f,  0.887701f,  0.988950f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cpow)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cpow test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cpow)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cpow test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cdiv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 2.239150f,  0.626800f,  0.113666f,  0.818037f,
      9.692400f,  1.705180f,  0.089300f,  2.206233f,
      1.329328f,  0.435140f,  0.996750f,  6.642100f,
      2.396850f,  1.849733f,  0.973366f,  8.948400f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cdiv)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cdiv)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cfmod_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.65443f,
      0.06924f,  0.35259f,  0.08037f,  0.06187f,
      0.23053f,  0.21757f,  0.59805f,  0.06421f,
      0.15874f,  0.25492f,  0.87603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cfmod)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cfmod test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cfmod)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cfmod test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_cremainder_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.0341f ,  0.65443f,
      0.06924f,  0.35259f,  0.08037f,  0.06187f,
      0.23053f,  0.21757f,  0.59805f,  0.06421f,
      0.15874f,  0.25492f,  0.87603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cremainder)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cremainder test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cremainder)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cremainder test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_addcmul_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.452624f,  0.091955f,  0.058117f,  0.873422f,
      0.994712f,  0.952386f,  0.331893f,  0.725865f,
      1.070150f,  0.263147f,  0.635882f,  0.687594f,
      1.008825f,  0.636304f,  0.920488f,  0.897575f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, addcmul)(frestnsr, ftnsr1c, 0.3f, ftnsr2c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addcmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_addcdiv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.567695f,  2.990260f,  0.300960f,  0.996606f,
      3.516450f,  1.251776f,  0.390893f,  1.372930f,
      1.215470f,  0.399880f,  0.703140f,  3.002620f,
      1.271775f,  1.459190f,  0.930916f,  1.168410f   };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, addcdiv)(frestnsr, ftnsr1c, 0.3f, ftnsr2c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addcdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_addmv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4[4] = {0.171163f,  0.574637f,  0.448013f,  0.634511f};
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4, 4), 1, size4, NULL);

  aiatensor_(float, addmv)(frestnsr, 0.2f, fvec2, 0.3f, ftnsr1c, fvec1);

  ck_assert_msg(aiatensor_(float, isVector)(frestnsr), "result should be a vector");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
    "addmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr);

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_addmm_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.289173f,  0.283463f,  0.571895f,  0.273271f,
      0.545708f,  0.499517f,  0.631792f,  0.445307f,
      0.372787f,  0.210716f,  0.509776f,  0.349444f,
      0.289440f,  0.292052f,  0.511183f,  0.368199f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // TODO: result with different strides

  aiatensor_(float, addmm)(frestnsr, 0.2f, ftnsr1c, 0.3f, ftnsr2c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // aiatensor_(float, addmm)(frestnsr, 0.2f, ftnsr1c, 0.3f, ftnsr2nc, ftnsr3c);
  // ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  // ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  // ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
  //   "addmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
  //   aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // aiatensor_(float, addmm)(frestnsr, 0.2f, ftnsr1c, 0.3f, ftnsr2c, ftnsr3nc);
  // ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  // ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  // ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
  //   "addmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
  //   aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_addr_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.102682f,  0.070977f,  0.056027f,  0.187398f,
      0.252874f,  0.433502f,  0.237507f,  0.386678f,
      0.218898f,  0.189616f,  0.242628f,  0.274122f,
      0.217981f,  0.227866f,  0.273621f,  0.291992f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, addr)(frestnsr, 0.2f, ftnsr1c, 0.3f, fvec1, fvec2);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_trace_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp = 1.399999f;
  float res = aiatensor_(float, trace)(ftnsr3c);
  ck_assert_msg(exp - res <= fepsi, "trace test failed. expected output = %f and actual output = %f", exp, res);
}
END_TEST

START_TEST(test_detpd_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp = 0.020788f;
  float res = aiatensor_(float, detpd)(fpdtnsrc);
  ck_assert_msg(exp - res <= fepsi, "detpd test failed. expected output = %f and actual output = %f", exp, res);
}
END_TEST

START_TEST(test_aIpX_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.74783f,  0.06268f,  0.03410f,  0.65443f,
      0.96924f,  1.15259f,  0.08037f,  0.66187f,
      0.93053f,  0.21757f,  0.89805f,  0.66421f,
      0.95874f,  0.55492f,  0.87603f,  1.19484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, aIpX)(frestnsr, ftnsr1c, 0.3f);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "aIpX test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_dot_float) {
  float res, exp;

  // vector-vector dot product
  exp = 1.707149f;
  res = aiatensor_(float, dot)(fvec1, fvec2);
  ck_assert_msg(fabsf(exp - res) <= fepsi, "dot test failed. expected output = %f and actual output = %f", exp, res);

  // matrix-matrix dot product
  exp = 5.094904f;
  res = aiatensor_(float, dot)(ftnsr2c, ftnsr1nc);
  ck_assert_msg(fabsf(exp - res) <= fepsi, "dot test failed. expected output = %f and actual output = %f", exp, res);
}
END_TEST

START_TEST(test_mv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4[4] = { 0.424800f,  1.266112f, 0.946628f, 1.487127f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4, 4), 1, size4, NULL);

  // contiguous tensor
  aiatensor_(float, mv)(frestnsr, ftnsr1c, fvec1);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "mv test 1 failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  // non-contiguous tensor
  aiatensor_(float, mv)(frestnsr, ftnsr1nc, fvec1);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "mv test 2 failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_sum_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x1[4] =
    { 1.199040f,  2.564070f,  2.410360f,  3.284530f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x1, 4), 2, size4x1, NULL);

  aiatensor_(float, sum)(frestnsr, ftnsr1c, 1); // sum along columns
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sum test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_sqrt_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.66920f, 0.25036f, 0.18466f, 0.80896f,
      0.98450f, 0.92335f, 0.28349f, 0.81355f,
      0.96464f, 0.46644f, 0.77333f, 0.81499f,
      0.97915f, 0.74493f, 0.93596f, 0.94596f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, sqrt)(frestnsr, ftnsr1c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "sqrt test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_exp_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 1.56491f, 1.06468f, 1.03468f, 1.92404f,
      2.63594f, 2.34571f, 1.08368f, 1.93841f,
      2.53585f, 1.24305f, 1.81856f, 1.94295f,
      2.60840f, 1.74180f, 2.40134f, 2.44694f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, exp)(frestnsr, ftnsr1c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "exp test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_log_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { -0.80334f,  -2.76971f,  -3.37845f,  -0.42399f,
      -0.03124f,  -0.15947f,  -2.52111f,  -0.41268f,
      -0.07200f,  -1.52523f,  -0.51408f,  -0.40915f,
      -0.04213f,  -0.58893f,  -0.13235f,  -0.11111f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, log)(frestnsr, ftnsr1c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "log test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_ceil_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f,
      1.00000f, 1.00000f, 3.00000f, 1.00000f,
      2.00000f, 1.00000f, 1.00000f, -3.00000f,
      1.00000f, -1.00000f, 1.00000f, 1.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, ceil)(frestnsr, ftnsr4c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "ceil test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_floor_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 2.00000f, 0.00000f,
      1.00000f, 0.00000f, 0.00000f, -4.00000f,
      0.00000f, -2.00000f, 0.00000f, 0.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, floor)(frestnsr, ftnsr4c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "floor test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_round_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.00000f, 0.00000f, 0.00000f, 1.00000f,
      1.00000f, 1.00000f, 2.00000f, 1.00000f,
      2.00000f, 0.00000f, 1.00000f, -4.00000f,
      1.00000f, -2.00000f, 1.00000f, 1.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, round)(frestnsr, ftnsr4c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "round test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_abs_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.44783f, 0.06268f, 0.03410f, 0.65443f,
      0.96924f, 0.85259f, 2.08037f, 0.66187f,
      1.93053f, 0.21757f, 0.59805f, 3.66421f,
      0.95874f, 1.55492f, 0.87603f, 0.89484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, abs)(frestnsr, ftnsr4c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "abs test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_trunc_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 2.00000f, 0.00000f,
      1.00000f, 0.00000f, 0.00000f, -3.00000f,
      0.00000f, -1.00000f,  0.00000f, 0.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, trunc)(frestnsr, ftnsr4c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "trunc test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_emulmv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.09790f, 0.06105f, 0.02796f, 0.61638f,
      0.21189f, 0.83043f, 0.06591f, 0.62339f,
      0.20342f, 0.21191f, 0.49047f, 0.62559f,
      0.20959f, 0.54050f, 0.71845f, 0.84282f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // contiguous tensor
  aiatensor_(float, emulmv)(frestnsr, ftnsr1c, fvec2);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "emulmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_eaddmv_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.66644f, 1.03669f, 0.85422f, 1.59629f,
      1.18785f, 1.82660f, 0.90049f, 1.60373f,
      1.14914f, 1.19158f, 1.41817f, 1.60607f,
      1.17735f, 1.52893f, 1.69615f, 1.83670f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // contiguous tensor
  aiatensor_(float, eaddmv)(frestnsr, ftnsr1c, fvec2);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "eaddmv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_mm_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.38482f, 1.08086f, 0.29282f, 0.53945f,
      1.13104f, 2.13600f, 1.17877f, 1.61364f,
      0.93393f, 1.83516f, 0.68607f, 1.53051f,
      1.50370f, 2.38014f, 1.10426f, 2.03384f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // TODO: result with different strides and non contiguous tensor

  aiatensor_(float, mm)(frestnsr, ftnsr1c, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "mm test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_fill_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, resize2d)(frestnsr, 4, 4);
  float exp4x4[16] =
    { 0.30000f, 0.30000f, 0.30000f, 0.30000f,
      0.30000f, 0.30000f, 0.30000f, 0.30000f,
      0.30000f, 0.30000f, 0.30000f, 0.30000f,
      0.30000f, 0.30000f, 0.30000f, 0.30000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, fill)(frestnsr, 0.3f);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "fill test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_zero_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, resize2d)(frestnsr, 4, 4);
  float exp4x4[16] =
    { 0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, zero)(frestnsr);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "zero test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_maskedFill_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, resize2d)(frestnsr, 4, 4);
  aiatensor_(float, zero)(frestnsr);
  float exp4x4[16] =
    { 0.00000f, 0.30000f, 0.00000f, 0.30000f,
      0.00000f, 0.30000f, 0.00000f, 0.30000f,
      0.00000f, 0.30000f, 0.00000f, 0.30000f,
      0.00000f, 0.30000f, 0.00000f, 0.30000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  unsigned char mask4x4[16] =
    { 0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1 };
  AIATensor(uchar) *masktnsr = aiatensor_(uchar, newFromData)(arr_(uchar, clone)(mask4x4, 16),
    2, size4x4, NULL);

  aiatensor_(float, maskedFill)(frestnsr, masktnsr, 0.3f);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "maskedFill test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(uchar, free)(masktnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_maskedCopy_float) {
  frestnsr = aiatensor_(float, empty)();
  aiatensor_(float, resize2d)(frestnsr, 4, 4);
  aiatensor_(float, zero)(frestnsr);
  float exp4x4[16] =
    { 0.00000f, 0.06268f, 0.00000f, 0.65443f,
      0.00000f, 0.85259f, 0.00000f, 0.66187f,
      0.00000f, 0.21757f, 0.00000f, 0.66421f,
      0.00000f, 0.55492f, 0.00000f, 0.89484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  unsigned char mask4x4[16] =
    { 0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1 };
  AIATensor(uchar) *masktnsr = aiatensor_(uchar, newFromData)(arr_(uchar, clone)(mask4x4, 16),
    2, size4x4, NULL);

  aiatensor_(float, maskedCopy)(frestnsr, masktnsr, ftnsr1c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "maskedCopy test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(uchar, free)(masktnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_zeros_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f,
      0.00000f, 0.00000f, 0.00000f, 0.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, zeros)(frestnsr, 2, size4x4, stride4x1);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "zeros test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_ones_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f,
      1.00000f, 1.00000f, 1.00000f, 1.00000f,
      1.00000f, 1.00000f, 1.00000f, 1.00000f,
      1.00000f, 1.00000f, 1.00000f, 1.00000f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, ones)(frestnsr, 2, size4x4, stride4x1);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "ones test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_xTAy_float) {
  float fres = 0.0f;
  float fexpres = 3.40940f;

  // TODO: x and y with different dimensions
  fres = aiatensor_(float, xTAy)(fvec1, ftnsr1c, fvec2);
  ck_assert_msg(epsieqf(fres, fexpres, fepsi),
    "xTAy test failed.\nexpected output =\n%0.5f\nactual output =\n%0.5f\n",
    fexpres, fres);
}
END_TEST

START_TEST(test_xTAx_float) {
  float fres = 0.0f;
  float fexpres = 2.29263f;

  fres = aiatensor_(float, xTAx)(fvec1, ftnsr1c);
  ck_assert_msg(epsieqf(fres, fexpres, fepsi),
    "xTAx test failed.\nexpected output =\n%0.5f\nactual output =\n%0.5f\n",
    fexpres, fres);
}
END_TEST

START_TEST(test_addbmm_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 0.27445f, 0.20270f, 0.32515f, 0.28704f,
      0.41147f, 0.42014f, 0.36989f, 0.31580f,
      0.45720f, 0.24477f, 0.56296f, 0.40146f,
      0.36945f, 0.27815f, 0.48286f, 0.46528f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  long size2x4x2[3] = {2l, 4l, 2l};
  AIATensor(float) *fbatch1 = aiatensor_(float, newFromData)(arr_(float, clone)(rndb4x4, 16), 3,
    size2x4x2, NULL);

  long size2x2x4[3] = {2l, 2l, 4l};
  AIATensor(float) *fbatch2 = aiatensor_(float, newFromData)(arr_(float, clone)(rndc4x4, 16), 3,
    size2x2x4, NULL);

  aiatensor_(float, addbmm)(frestnsr, 0.3f, ftnsr1c, 0.2f, fbatch1, fbatch2);
  ck_assert_msg(aiatensor_(float, isMatrix)(frestnsr), "result should be a matrix");
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addbmm test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fbatch1);
  aiatensor_(float, free)(fbatch2);
  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_xTApdIx_float) {
  float fres = 0.0f;
  float fexpres = 1.45543f;

  /* evaluate cholskey decomposition first */
  AIATensor(float) *fachol = aiatensor_(float, emptyAs)(fpdtnsrc);
  aiatensor_(float, potrf)(fachol, fpdtnsrc, "L");

  fres = aiatensor_(float, xTApdIx)(fvec1, fachol, "L");
  ck_assert_msg(epsieqf(fres, fexpres, fepsi),
    "xTApdIx test failed.\nexpected output =\n%0.5f\nactual output =\n%0.5f\n",
    fexpres, fres);

  aiatensor_(float, free)(fachol);
}
END_TEST

START_TEST(test_xTAsymmIy_float) {
  float fres = 0.0f;
  float fexpres = 1.07322f;

  /* evaluate cholskey decomposition first */
  AIATensor(float) *fachol = aiatensor_(float, emptyAs)(fpdtnsrc);
  aiatensor_(float, potrf)(fachol, fpdtnsrc, "L");

  fres = aiatensor_(float, xTAsymmIy)(fvec1, fachol, "L", fvec2);
  ck_assert_msg(epsieqf(fres, fexpres, fepsi),
    "xTAsymmIy test failed.\nexpected output =\n%0.5f\nactual output =\n%0.5f\n",
    fexpres, fres);

  aiatensor_(float, free)(fachol);
}
END_TEST

START_TEST(test_XTApdIXpaY_float) {
  frestnsr = aiatensor_(float, empty)();
  float exp4x4[16] =
    { 1.023968f,  0.292746f,  0.080034f,  0.273763f ,
      0.254709f,  1.199615f,  0.279474f,  0.213314f,
      0.199434f,  0.091158f,  1.063052f,  0.233864f,
      0.125236f,  0.271277f,  0.049421f,  1.027334f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  /* evaluate cholskey decomposition first */
  AIATensor(float) *fachol = aiatensor_(float, emptyAs)(fpdtnsrc);
  aiatensor_(float, potrf)(fachol, fpdtnsrc, "L");

  aiatensor_(float, XTApdIXpaY)(frestnsr, ftnsr1c, fachol, "L", 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(frestnsr, fexptnsr), "result has wrong dimensions");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "xTApdIXpaY test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, toString)(fexptnsr), aiatensor_(float, toString)(frestnsr));

  aiatensor_(float, free)(fachol);
  aiatensor_(float, free)(fexptnsr);
  aiatensor_(float, free)(frestnsr);
}
END_TEST


Suite *make_tensormath_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("TensorMath");
  tc = tcase_create("TensorMath");

  tcase_add_checked_fixture(tc, tensormath_setup, tensormath_teardown);

  tcase_add_test(tc, test_add_float);
  tcase_add_test(tc, test_sub_float);
  tcase_add_test(tc, test_mul_float);
  tcase_add_test(tc, test_div_float);
  tcase_add_test(tc, test_fmod_float);
  tcase_add_test(tc, test_remainder_float);
  tcase_add_test(tc, test_clamp_float);
  tcase_add_test(tc, test_cadd_float);
  tcase_add_test(tc, test_csub_float);
  tcase_add_test(tc, test_cmul_float);
  tcase_add_test(tc, test_cpow_float);
  tcase_add_test(tc, test_cdiv_float);
  tcase_add_test(tc, test_cfmod_float);
  tcase_add_test(tc, test_cremainder_float);
  tcase_add_test(tc, test_addcmul_float);
  tcase_add_test(tc, test_addcdiv_float);
  tcase_add_test(tc, test_addmv_float);
  tcase_add_test(tc, test_addmm_float);
  tcase_add_test(tc, test_addr_float);
  tcase_add_test(tc, test_trace_float);
  tcase_add_test(tc, test_detpd_float);
  tcase_add_test(tc, test_aIpX_float);
  tcase_add_test(tc, test_dot_float);
  tcase_add_test(tc, test_mv_float);
  tcase_add_test(tc, test_sum_float);
  tcase_add_test(tc, test_sqrt_float);
  tcase_add_test(tc, test_exp_float);
  tcase_add_test(tc, test_log_float);
  tcase_add_test(tc, test_ceil_float);
  tcase_add_test(tc, test_floor_float);
  tcase_add_test(tc, test_round_float);
  tcase_add_test(tc, test_abs_float);
  tcase_add_test(tc, test_trunc_float);
  tcase_add_test(tc, test_emulmv_float);
  tcase_add_test(tc, test_eaddmv_float);
  tcase_add_test(tc, test_mm_float);
  tcase_add_test(tc, test_fill_float);
  tcase_add_test(tc, test_zero_float);
  tcase_add_test(tc, test_maskedFill_float);
  tcase_add_test(tc, test_maskedCopy_float);
  tcase_add_test(tc, test_zeros_float);
  tcase_add_test(tc, test_ones_float);
  tcase_add_test(tc, test_xTAy_float);
  tcase_add_test(tc, test_xTAx_float);
  tcase_add_test(tc, test_addbmm_float);
  tcase_add_test(tc, test_xTApdIx_float);
  tcase_add_test(tc, test_xTAsymmIy_float);
  tcase_add_test(tc, test_XTApdIXpaY_float);

  suite_add_tcase(s, tc);
  return s;
}