#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aianon/tensor/tensor.h>

#include "sample_tensors.h"

AIATensor(float) *ftnsr;

void tensor_setup(void) {
  ftnsr = mktnsr_(float, rnd4x4)();
}

void tensor_teardown(void) {
  aiatensor_(float, free)(ftnsr);
}

START_TEST(test_tensor_create_from_data) {
  ck_assert_msg(aiatensor_(float, size)(ftnsr, 0) == 4,
    "wrong size at dim 0 = %li\n", aiatensor_(float, size)(ftnsr, 0));
  ck_assert_msg(aiatensor_(float, size)(ftnsr, 1) == 4,
    "wrong size at dim 1 = %li\n", aiatensor_(float, size)(ftnsr, 1));
}
END_TEST

Suite *make_tensor_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("Tensor");
  tc = tcase_create("Tensor");

  tcase_add_checked_fixture(tc, tensor_setup, tensor_teardown);
  tcase_add_test(tc, test_tensor_create_from_data);
  suite_add_tcase(s, tc);
  return s;
}