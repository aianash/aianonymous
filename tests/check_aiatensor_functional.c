#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiatensor/functional.h>

static float rnd4x4[16] =
  { 1.f,   2.f,   3.f,   4.f,
    5.f,   6.f,   7.f,   8.f,
    9.f,   10.f,  11.f,  12.f,
    13.f,  14.f,  15.f,  16.f };

static float rnd4x4T[16] =
  { 0.079918f,  0.849078f,  0.664866f,  0.417380f,
    0.975864f,  0.665310f,  0.303851f,  0.904275f,
    0.266865f,  0.931576f,  0.210187f,  0.164664f,
    0.912477f,  0.711065f,  0.779473f,  0.091194f };

static float vector[4] =
  {1.f, 2.f, 3.f, 4.f};

static long size4x4[2] = {4l, 4l};
static long size4[1] = {4l};

static AIATensor(float) *f4x4tnsr;
static AIATensor(float) *f4x4tnsrT;
static AIATensor(float) *vectorT;

void functional_setup(void) {
  f4x4tnsr = aiatensor_(float, newFromData)(arr_(float, clone)(rnd4x4, 16), 2, size4x4, NULL);
  f4x4tnsrT = aiatensor_(float, newFromData)(arr_(float, clone)(rnd4x4T, 16), 2, size4x4, NULL);
  vectorT = aiatensor_(float, newFromData)(arr_(float, clone)(vector, 4), 1, size4, NULL);
}

void functional_teardown(void) {
  aiatensor_(float, free)(f4x4tnsr);
  aiatensor_(float, free)(f4x4tnsrT);
  aiatensor_(float, free)(vectorT);
}

//
START_TEST(test_mforeach) {
  float *ele = NULL;
  long count = 0;

  mforeach(ele, f4x4tnsr) {
    count++;
  }
  endmforeach()
  ck_assert_msg(count == 16, "Failed mforeach count = %ld", count);
}
END_TEST

//
START_TEST(test_mzip) {
  float *ele1 = NULL;
  float *ele2 = NULL;
  float *ele3 = NULL;
  long count = 0;

  count = 0;
  mzip2(ele1, f4x4tnsr, ele2, f4x4tnsrT) {
    count++;
  }
  endmzip2
  ck_assert_msg(count == 16, "Failed mzip2 count = %ld", count);

  AIATensor(float) *tnsr3 = f4x4tnsr;
  count = 0;
  mzip3(ele1, f4x4tnsr, ele2, f4x4tnsrT, ele3, tnsr3) {
    count++;
  }
  endmzip3
  ck_assert_msg(count == 16, "Failed mzip3 count = %ld", count);
}
END_TEST

//
START_TEST(test_mvzip) {
  float *ele1 = NULL;
  float *ele2 = NULL;
  long count = 0;

  mvrzip(ele1, f4x4tnsr, ele2, vectorT) {
    count++;
  }
  endmvrzip()
  ck_assert_msg(count == 16, "Failed mvrzip count = %ld", count);

  count = 0;
  mvczip(ele1, f4x4tnsr, ele2, vectorT) {
    count++;
  }
  endmvczip()
  ck_assert_msg(count == 16, "Failed mvczip count = %ld", count);
}
END_TEST

//
START_TEST(test_mfor) {
  float *ele1 = NULL;
  float *ele2 = NULL;

  long count = 0;
  mfor(ele1, f4x4tnsr, ele2, f4x4tnsrT) {
    count++;
  }
  endmfor()

  ck_assert_msg(count == 256, "Failed mfor count = %ld", count);
}
END_TEST

//
START_TEST(test_vforeach) {
  float *ele = NULL;
  long count = 0;

  vforeach(ele, vectorT) {
    count++;
  }
  endvforeach()
  ck_assert_msg(count == 4, "Failed vforeach count = %ld", count);
}
END_TEST

//
START_TEST(test_vzip) {
  float *ele1 = NULL;
  float *ele2 = NULL;
  float *ele3 = NULL;
  long count = 0;
  AIATensor(float) *vector2 = vectorT;

  vzip2(ele1, vectorT, ele2, vector2) {
    count++;
  }
  endvzip2
  ck_assert_msg(count == 4, "Failed vzip2 count = %ld", count);

  AIATensor(float) *vector3 = vectorT;

  count = 0;
  vzip3(ele1, vectorT, ele2, vector2, ele3, vector3) {
    count++;
  }
  endvzip3
  ck_assert_msg(count == 4, "Failed vzip3 count = %ld", count);
}
END_TEST

//
START_TEST(test_vfor) {
  float *ele1 = NULL;
  float *ele2 = NULL;
  long count = 0;
  AIATensor(float) *vector2 = vectorT;

  vfor(ele1, vectorT, ele2, vector2) {
    count++;
  }
  endvfor()
  ck_assert_msg(count == 16, "Failed vfor count = %ld", count);
}
END_TEST

//
Suite *make_functional_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("Functional");
  tc = tcase_create("Functional");

  tcase_add_checked_fixture(tc, functional_setup, functional_teardown);
  tcase_add_test(tc, test_mforeach);
  tcase_add_test(tc, test_mzip);
  tcase_add_test(tc, test_mfor);
  tcase_add_test(tc, test_vforeach);
  tcase_add_test(tc, test_vzip);
  tcase_add_test(tc, test_vfor);
  tcase_add_test(tc, test_mvzip);

  suite_add_tcase(s, tc);
  return s;
}