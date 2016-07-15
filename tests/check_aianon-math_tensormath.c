#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/math.h>

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

static long size4x4[2] = {4l, 4l};

static float fepsi = 1e-5f;

// contiguous tensors
AIATensor(float) *ftnsr1c;
AIATensor(float) *ftnsr2c;
AIATensor(float) *ftnsr3c;

// transposed tensors
AIATensor(float) *ftnsr1cT;
AIATensor(float) *ftnsr2cT;
AIATensor(float) *ftnsr3cT;

// non-contiguous tensors
AIATensor(float) *ftnsr1nc;
AIATensor(float) *ftnsr2nc;
AIATensor(float) *ftnsr3nc;

// result tensors
AIATensor(float) *frestnsr;
AIATensor(float) *fexptnsr;

void tensormath_setup(void) {
  ftnsr1c  = aiatensor_(float, newFromData)(arr_(float, clone)(rnda4x4, 16), 2, size4x4, NULL);
  ftnsr2c  = aiatensor_(float, newFromData)(arr_(float, clone)(rndb4x4, 16), 2, size4x4, NULL);
  ftnsr3c  = aiatensor_(float, newFromData)(arr_(float, clone)(rndc4x4, 16), 2, size4x4, NULL);
  ftnsr1cT = aiatensor_(float, newFromData)(arr_(float, clone)(rnda4x4T, 16), 2, size4x4, NULL);
  ftnsr1nc = aiatensor_(float, empty)();
  aiatensor_(float, transpose)(ftnsr1nc, ftnsr1cT, 0, 1);
  frestnsr = aiatensor_(float, empty)();
}

void tensormath_teardown(void) {
  aiatensor_(float, free)(ftnsr1c);
  aiatensor_(float, free)(ftnsr2c);
  aiatensor_(float, free)(ftnsr3c);
  aiatensor_(float, free)(ftnsr1cT);
  aiatensor_(float, free)(ftnsr1nc);
  aiatensor_(float, free)(frestnsr);
}

START_TEST(test_add_float) {
  aiatensor_(float, add)(frestnsr, ftnsr1c, 0.8327f);
  float exp4x4[16] =
    { 1.28053f,  0.89538f,  0.86680f,  1.48713f,
      1.80194f,  1.68529f,  0.91307f,  1.49457f,
      1.76323f,  1.05027f,  1.43075f,  1.49691f,
      1.79144f,  1.38762f,  1.70873f,  1.72754f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "add test failed %d and %d", frestnsr->size[0], frestnsr->size[1]);

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_sub_float) {
  aiatensor_(float, sub)(frestnsr, ftnsr1c, 0.8327f);
  float exp4x4[16] =
    { -0.38487f, -0.77002f, -0.79860f, -0.17827f,
       0.13654f,  0.01989f, -0.75233f, -0.17083f,
       0.09783f, -0.61513f, -0.23465f, -0.16849f,
       0.12604f, -0.27778f,  0.04333f,  0.06214f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "sub test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_mul_float) {
  aiatensor_(float, mul)(frestnsr, ftnsr1c, 0.8f);
  float exp4x4[16] =
    { 0.358264f,  0.050144f,  0.027280f,  0.523544f,
      0.775392f,  0.682072f,  0.064296f,  0.529496f,
      0.744424f,  0.174056f,  0.478440f,  0.531368f,
      0.766992f,  0.443936f,  0.700824f,  0.715872f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "mul test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_div_float) {
  aiatensor_(float, div)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 2.23915f,  0.31340f,  0.17050f,  3.27215f,
      4.84620f,  4.26295f,  0.40185f,  3.30935f,
      4.65265f,  1.08785f,  2.99025f,  3.32105f,
      4.79370f,  2.77460f,  4.38015f,  4.47420f  };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "div test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_fmod_float) {
  aiatensor_(float, fmod)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.05443f,
      0.16924f,  0.05259f,  0.08037f,  0.06187f,
      0.13053f,  0.01757f,  0.19805f,  0.06421f,
      0.15874f,  0.15492f,  0.07603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "fmod test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_remainder_float) {
  aiatensor_(float, remainder)(frestnsr, ftnsr1c, 0.2f);
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.05443f,
      0.16924f,  0.05259f,  0.08037f,  0.06187f,
      0.13053f,  0.01757f,  0.19805f,  0.06421f,
      0.15874f,  0.15492f,  0.07603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "remainder test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_clamp_float) {
  aiatensor_(float, clamp)(frestnsr, ftnsr1c, 0.17328f, 0.51223f);
  float exp4x4[16] =
    { 0.44783f,  0.17328f,  0.17328f,  0.51223f,
      0.51223f,  0.51223f,  0.17328f,  0.51223f,
      0.51223f,  0.21757f,  0.51223f,  0.51223f,
      0.51223f,  0.51223f,  0.51223f,  0.51223f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi), "clamp test failed");

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cadd_float) {
  float exp4x4[16] =
    { 0.471803f,  0.355438f,  0.114158f,  0.928171f,
      1.223961f,  1.052183f,  0.359841f,  0.875188f,
      1.129988f,  0.308725f,  0.661104f,  0.898051f,
      1.083954f,  0.826201f,  0.925428f,  0.922197f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when ftnsr1c and frestnsr are contiguous and different
  aiatensor_(float, cadd)(frestnsr, ftnsr1c, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when ftnsr1c and frestnsr are contiguous and same
  aiatensor_(float, copy)(frestnsr, ftnsr1c);
  aiatensor_(float, cadd)(frestnsr, frestnsr, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cadd)(frestnsr, ftnsr1nc, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_csub_float) {
  float exp4x4[16] =
    { 0.423857, -0.230078, -0.045958,  0.380689,
      0.714519,  0.652997, -0.199101,  0.448552,
      0.731072,  0.126415,  0.534996,  0.430369,
      0.833526,  0.283639,  0.826632,  0.867483 };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when ftnsr1c and frestnsr are contiguous and different
  aiatensor_(float, csub)(frestnsr, ftnsr1c, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "csub test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when ftnsr1c and frestnsr are contiguous and same
  aiatensor_(float, copy)(frestnsr, ftnsr1c);
  aiatensor_(float, csub)(frestnsr, frestnsr, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "csub test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, csub)(frestnsr, ftnsr1nc, 0.3f, ftnsr2c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cadd test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cmul_float) {
  float exp4x4[16] =
    { 0.089566f,  0.006268f,  0.010230f,  0.523544f,
      0.096924f,  0.426295f,  0.072333f,  0.198561f,
      0.651371f,  0.108785f,  0.358830f,  0.066421f,
      0.383496f,  0.166476f,  0.788427f,  0.089484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cmul)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cmul)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cpow_float) {
  float exp4x4[16] =
    { 0.851574f,  0.758076f,  0.362932f,  0.712345f,
      0.996880f,  0.923358f,  0.103415f,  0.883551f,
      0.950848f,  0.466443f,  0.734585f,  0.959910f,
      0.983287f,  0.838048f,  0.887701f,  0.988950f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cpow)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cpow test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cpow)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cpow test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cdiv_float) {
  float exp4x4[16] =
    { 2.239150f,  0.626800f,  0.113666f,  0.818037f,
      9.692400f,  1.705180f,  0.089300f,  2.206233f,
      1.329328f,  0.435140f,  0.996750f,  6.642100f,
      2.396850f,  1.849733f,  0.973366f,  8.948400f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cdiv)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cdiv)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cfmod_float) {
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.03410f,  0.65443f,
      0.06924f,  0.35259f,  0.08037f,  0.06187f,
      0.23053f,  0.21757f,  0.59805f,  0.06421f,
      0.15874f,  0.25492f,  0.87603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cfmod)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cfmod test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cfmod)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cfmod test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_cremainder_float) {
  float exp4x4[16] =
    { 0.04783f,  0.06268f,  0.0341f ,  0.65443f,
      0.06924f,  0.35259f,  0.08037f,  0.06187f,
      0.23053f,  0.21757f,  0.59805f,  0.06421f,
      0.15874f,  0.25492f,  0.87603f,  0.09484f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  // when tensors are contiguous
  aiatensor_(float, cremainder)(frestnsr, ftnsr1c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cremainder test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  // when tensors are not contiguous
  aiatensor_(float, cremainder)(frestnsr, ftnsr1nc, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "cremainder test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_addcmul_float) {
  float exp4x4[16] =
    { 0.452624f,  0.091955f,  0.058117f,  0.873422f,
      0.994712f,  0.952386f,  0.331893f,  0.725865f,
      1.070150f,  0.263147f,  0.635882f,  0.687594f,
      1.008825f,  0.636304f,  0.920488f,  0.897575f };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, addcmul)(frestnsr, ftnsr1c, 0.3f, ftnsr2c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addcmul test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
}
END_TEST

START_TEST(test_addcdiv_float) {
  float exp4x4[16] =
    { 0.567695f,  2.990260f,  0.300960f,  0.996606f,
      3.516450f,  1.251776f,  0.390893f,  1.372930f,
      1.215470f,  0.399880f,  0.703140f,  3.002620f,
      1.271775f,  1.459190f,  0.930916f,  1.168410f   };
  fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp4x4, 16), 2, size4x4, NULL);

  aiatensor_(float, addcdiv)(frestnsr, ftnsr1c, 0.3f, ftnsr2c, ftnsr3c);
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 0) == 4, "result has wrong dim 0");
  ck_assert_msg(aiatensor_(float, size)(frestnsr, 1) == 4, "result has wrong dim 1");
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "addcdiv test failed.\nexpected output =\n%s\nactual output =\n%s\n",
    aiatensor_(float, mat2str)(fexptnsr), aiatensor_(float, mat2str)(frestnsr));

  aiatensor_(float, free)(fexptnsr);
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

  suite_add_tcase(s, tc);
  return s;
}