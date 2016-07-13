#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aianon/tensor/tensor.h>

AIATensor(float) *tnsr;

void setup(void) {
  tnsr = aiatensor_(float, empty)();
}

void teardown(void) {
  aiatensor_(float, free)(tnsr);
}

START_TEST(test_tensor_create)
{
  aiatensor_(float, resize1d)(tnsr, 10);
  ck_assert_int_eq(aiatensor_(float, size)(tnsr, 0), 10);
}
END_TEST

Suite * tensor_suite(void) {
  Suite *s;
  TCase *tc_tensor;

  s = suite_create("AIAnon");
  tc_tensor = tcase_create("Tensor");

  tcase_add_checked_fixture(tc_tensor, setup, teardown);
  tcase_add_test(tc_tensor, test_tensor_create);
  suite_add_tcase(s, tc_tensor);

  return s;
}

int main(void) {

  int number_failed;
  Suite *s;
  SRunner *sr;

  s = tensor_suite();
  sr = srunner_create(s);

  srunner_run_all(sr, CK_NORMAL);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);
  return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}